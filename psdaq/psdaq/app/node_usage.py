#!/usr/bin/env python3
"""
CNF Parser - Configuration file analyzer for LCLS2 DAQ

Parses procmgr CNF files (.cnf or .py) to show resource usage,
available nodes, lane allocations, and process summaries.

Usage:
    cnf_parser.py <file1.cnf> [file2.py ...]
    cnf_parser.py --help
"""

import os
import sys
import copy
import re
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Set, Tuple


class CnfParser:
    """Parser for procmgr CNF configuration files"""

    def __init__(self, cnf_file: str, follow_imports: bool = True):
        self.cnf_file = Path(cnf_file).resolve()
        self.follow_imports = follow_imports

        # Parsed data
        self.platform = None
        self.station = None
        self.imports = []
        self.procmgr_config = []
        self.variables = {}
        self.config_obj = None  # If Config() is used

        # Transformation tracking
        self.renames = {}      # {new_id: old_id}
        self.extended = set()  # IDs that were added via extend()

        # Parse the file
        self.parse()

    def parse(self):
        """Parse the CNF file and extract configuration"""
        if not self.cnf_file.exists():
            raise FileNotFoundError(f"Config file not found: {self.cnf_file}")

        # Create namespace for execution
        namespace = self._create_namespace()

        # Add the file's directory to sys.path for imports
        file_dir = str(self.cnf_file.parent)
        if file_dir not in sys.path:
            sys.path.insert(0, file_dir)

        try:
            # Execute the CNF file
            with open(self.cnf_file, 'r') as f:
                code = compile(f.read(), str(self.cnf_file), 'exec')
                exec(code, namespace, namespace)

            # Extract results
            self.platform = namespace.get('platform', None)
            self.station = namespace.get('station', None)

            # Check if Config() was used
            if 'config' in namespace:
                config_obj = namespace['config']
                if hasattr(config_obj, 'select_config'):
                    # Config object was used
                    self.config_obj = config_obj
                    self.procmgr_config = list(config_obj.select_config.values())
                    # Detect renames and extensions
                    self._detect_transformations(config_obj)
                elif isinstance(config_obj, list):
                    # config is a list
                    self.procmgr_config = config_obj
            elif 'procmgr_config' in namespace:
                self.procmgr_config = namespace['procmgr_config']

            # Store other interesting variables
            for key, value in namespace.items():
                if not key.startswith('_') and key not in ['procmgr_config', 'config', 'platform', 'station']:
                    if isinstance(value, (str, int, float, list, dict)):
                        self.variables[key] = value

        finally:
            # Clean up sys.path
            if file_dir in sys.path:
                sys.path.remove(file_dir)

    def _create_namespace(self) -> Dict[str, Any]:
        """Create namespace for executing CNF file"""
        namespace = {
            '__builtins__': __builtins__,
            '__file__': str(self.cnf_file),
            '__name__': '__main__',
            'platform': None,
            'station': None,
            'procmgr_config': None,
        }

        # Add common environment variables
        namespace['CONDA_PREFIX'] = os.environ.get('CONDA_PREFIX', '')
        namespace['CONFIGDIR'] = str(self.cnf_file.parent)

        # Import Config class if needed
        try:
            from psdaq.slurm.config import Config
            namespace['Config'] = Config
        except ImportError:
            pass

        # Add common dictionary keys as variables for convenience
        namespace['host'] = 'host'
        namespace['cores'] = 'cores'
        namespace['id'] = 'id'
        namespace['flags'] = 'flags'
        namespace['env'] = 'env'
        namespace['cmd'] = 'cmd'
        namespace['rtprio'] = 'rtprio'
        namespace['port'] = 'port'

        return namespace

    def get_processes(self) -> List[Dict[str, Any]]:
        """Get the final list of processes with defaults filled in"""
        processes = []
        for proc in self.procmgr_config:
            # Make a copy and fill in defaults
            p = copy.deepcopy(proc)
            if 'host' not in p:
                p['host'] = 'localhost'
            processes.append(p)
        return processes

    def _detect_transformations(self, config_obj):
        """Detect which processes were renamed or extended"""
        if not hasattr(config_obj, 'main_config') or not hasattr(config_obj, 'select_config'):
            return

        main_config = config_obj.main_config      # {id: {...}} - original base config
        select_config = config_obj.select_config  # {id: {...}} - final transformed config

        # Build cmd -> id mapping from main_config for detecting renames
        # Use (host, cmd) tuple as key for more accurate matching
        main_by_cmd = {}
        for cid, cfg in main_config.items():
            key = (cfg.get('host', 'localhost'), cfg.get('cmd', ''))
            main_by_cmd[key] = cid

        for new_id, cfg in select_config.items():
            if new_id in main_config:
                # Direct select, no rename - it exists with same ID
                pass
            else:
                # Not in main_config under this ID - check if renamed or extended
                key = (cfg.get('host', 'localhost'), cfg.get('cmd', ''))
                if key in main_by_cmd:
                    # Command matches something in main_config -> this was renamed
                    old_id = main_by_cmd[key]
                    self.renames[new_id] = old_id
                else:
                    # No match -> this was extended (added new)
                    self.extended.add(new_id)


class ResourceAnalyzer:
    """Analyzes resource usage from parsed configurations"""

    def __init__(self, parsers: List[CnfParser]):
        self.parsers = parsers
        self.processes = []
        self.renames = {}      # {new_id: old_id} - aggregated from all parsers
        self.extended = set()  # IDs that were extended - aggregated from all parsers

        # Collect all processes and transformations from all parsers
        for parser in parsers:
            self.processes.extend(parser.get_processes())
            self.renames.update(parser.renames)
            self.extended.update(parser.extended)

    def get_hosts(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group processes by host"""
        hosts = defaultdict(list)
        for proc in self.processes:
            host = proc.get('host', 'localhost')
            hosts[host].append(proc)
        return dict(hosts)

    def get_lane_usage(self) -> List[Tuple[str, str, str, str]]:
        """Parse lane usage from commands

        Returns: List of (host, device, lane_mask_hex, process_id) tuples
        """
        lane_usage = []

        for proc in self.processes:
            host = proc.get('host', 'localhost')
            proc_id = proc.get('id', 'unknown')
            cmd = proc.get('cmd', '')

            # Parse device (e.g., /dev/datadev_0 or /dev/datadev_1)
            device_match = re.search(r'/dev/(datadev_\d+)', cmd)
            if device_match:
                device = device_match.group(1)

                # Parse lane mask (e.g., -l 0x1 or -l 0x4)
                lane_match = re.search(r'-l\s+(0x[0-9a-fA-F]+)', cmd)
                if lane_match:
                    lane_mask = lane_match.group(1)  # Keep as hex string
                    lane_usage.append((host, device, lane_mask, proc_id))

        # Sort by host, device, then lane
        lane_usage.sort(key=lambda x: (x[0], x[1], int(x[2], 16)))
        return lane_usage

    def _mask_to_lanes(self, mask: int) -> List[int]:
        """Convert lane mask to list of lane numbers"""
        lanes = []
        for i in range(32):  # Check up to 32 lanes
            if mask & (1 << i):
                lanes.append(i)
        return lanes

    def get_cores_usage(self) -> Dict[str, int]:
        """Get core allocation by host"""
        cores = defaultdict(int)
        for proc in self.processes:
            host = proc.get('host', 'localhost')
            proc_cores = proc.get('cores', 0)
            if isinstance(proc_cores, int):
                cores[host] += proc_cores
        return dict(cores)

    def get_process_types(self) -> Dict[str, List[str]]:
        """Categorize processes by type"""
        types = defaultdict(list)

        for proc in self.processes:
            proc_id = proc.get('id', 'unknown')
            cmd = proc.get('cmd', '')

            # Categorize based on command or ID
            if 'control' in proc_id or 'teb' in proc_id or 'meb' in proc_id:
                types['Control/Infrastructure'].append(proc_id)
            elif 'ami' in proc_id.lower():
                types['AMI'].append(proc_id)
            elif 'timing' in proc_id or 'bld' in proc_id:
                types['Timing/BLD'].append(proc_id)
            elif any(cam in cmd for cam in ['drp_pva', 'pva_cmd', 'opal', 'piranha', 'andor', 'archon']):
                if 'pva' in cmd:
                    types['Cameras (PVA)'].append(proc_id)
                else:
                    types['Cameras (Native)'].append(proc_id)
            elif 'wave8' in cmd or 'w8' in proc_id:
                types['Wave8 Digitizers'].append(proc_id)
            elif 'hsd' in proc_id.lower():
                types['HSD'].append(proc_id)
            elif 'encoder' in proc_id:
                types['Encoders'].append(proc_id)
            elif 'epics' in proc_id or 'epicsArch' in cmd:
                types['EPICS Archive'].append(proc_id)
            elif 'procstat' in proc_id or 'daqstat' in proc_id:
                types['Monitoring'].append(proc_id)
            else:
                types['Other'].append(proc_id)

        return dict(types)


class OutputFormatter:
    """Formats output for display"""

    def __init__(self, parsers: List[CnfParser], analyzer: ResourceAnalyzer, verbose: bool = False):
        self.parsers = parsers
        self.analyzer = analyzer
        self.verbose = verbose

    def format_summary(self) -> str:
        """Generate the main summary output"""
        lines = []

        # Header
        lines.append("CNF Parser - Resource Usage Summary")
        lines.append("=" * 80)

        # Source files
        if len(self.parsers) == 1:
            lines.append(f"Source: {self.parsers[0].cnf_file.name}")
            if self.parsers[0].platform:
                lines.append(f"Platform: {self.parsers[0].platform}")
        else:
            lines.append(f"Sources: {', '.join(p.cnf_file.name for p in self.parsers)}")

        total_procs = len(self.analyzer.processes)
        lines.append(f"Total Processes: {total_procs}")
        lines.append("")

        # Verbose: detailed process table
        if self.verbose:
            lines.append(self._format_detailed_process_table())
            lines.append("")

        # Host allocation
        lines.append(self._format_host_section())
        lines.append("")

        # Lane usage
        lane_usage = self.analyzer.get_lane_usage()
        if lane_usage:
            lines.append(self._format_lane_section(lane_usage))
            lines.append("")

        # Process types
        lines.append(self._format_process_types())
        lines.append("")

        # Transformations (renames and extensions) - only if any exist
        if self.analyzer.renames or self.analyzer.extended:
            lines.append(self._format_transformations())
            lines.append("")

        # Quick stats
        lines.append(self._format_quick_stats())

        return "\n".join(lines)

    def _format_host_section(self) -> str:
        """Format the host allocation section"""
        lines = []
        lines.append("┌" + "─" * 78 + "┐")
        lines.append("│ " + "Host Resource Allocation".ljust(77) + "│")
        lines.append("├" + "─" * 20 + "┬" + "─" * 7 + "┬" + "─" * 50 + "┤")
        lines.append("│ " + "Host".ljust(19) + "│ " + "Procs".ljust(6) + "│ " + "Process IDs".ljust(49) + "│")
        lines.append("├" + "─" * 20 + "┼" + "─" * 7 + "┼" + "─" * 50 + "┤")

        hosts = self.analyzer.get_hosts()
        cores_usage = self.analyzer.get_cores_usage()

        for host in sorted(hosts.keys()):
            procs = hosts[host]
            proc_count = len(procs)

            # Get process IDs
            proc_ids = [p.get('id', 'unknown') for p in procs]
            proc_list = ', '.join(proc_ids)

            # Word wrap the process list if it's too long
            first_line = True
            remaining = proc_list
            while remaining:
                if len(remaining) <= 49:
                    # Fits on one line
                    if first_line:
                        lines.append(f"│ {host[:19].ljust(19)}│ {str(proc_count).rjust(5)} │ {remaining.ljust(49)}│")
                        first_line = False
                    else:
                        lines.append(f"│ {''.ljust(19)}│ {''.ljust(6)}│ {remaining.ljust(49)}│")
                    break
                else:
                    # Need to wrap - find last comma before 49 chars
                    wrap_pos = remaining[:49].rfind(',')
                    if wrap_pos == -1:
                        # No comma found, just split at 49
                        wrap_pos = 49
                    else:
                        wrap_pos += 1  # Include the comma

                    chunk = remaining[:wrap_pos].rstrip(', ')
                    if first_line:
                        lines.append(f"│ {host[:19].ljust(19)}│ {str(proc_count).rjust(5)} │ {chunk.ljust(49)}│")
                        first_line = False
                    else:
                        lines.append(f"│ {''.ljust(19)}│ {''.ljust(6)}│ {chunk.ljust(49)}│")

                    # Continue with remainder
                    remaining = remaining[wrap_pos:].lstrip(', ')

            # Add cores info if available
            if host in cores_usage and cores_usage[host] > 0:
                cores_info = f"  Cores: {cores_usage[host]} allocated"
                lines.append(f"│ {''.ljust(19)}│ {''.ljust(6)}│ {cores_info[:49].ljust(49)}│")

        lines.append("└" + "─" * 20 + "┴" + "─" * 7 + "┴" + "─" * 50 + "┘")

        return "\n".join(lines)

    def _format_detailed_process_table(self) -> str:
        """Format detailed process table for verbose mode"""
        lines = []
        lines.append("┌" + "─" * 118 + "┐")
        lines.append("│ " + "Detailed Process List".ljust(117) + "│")
        lines.append("├" + "─" * 20 + "┬" + "─" * 20 + "┬" + "─" * 7 + "┬" + "─" * 6 + "┬" + "─" * 62 + "┤")
        lines.append("│ " + "Host".ljust(19) + "│ " + "ID".ljust(19) + "│ " + "Port".ljust(6) + "│ " + "Flags".ljust(5) + "│ " + "Command".ljust(61) + "│")
        lines.append("├" + "─" * 20 + "┼" + "─" * 20 + "┼" + "─" * 7 + "┼" + "─" * 6 + "┼" + "─" * 62 + "┤")

        # Sort processes by host, then by id
        sorted_procs = sorted(self.analyzer.processes, key=lambda p: (p.get('host', 'localhost'), p.get('id', '')))

        prev_host = None
        for proc in sorted_procs:
            host = proc.get('host', 'localhost')
            proc_id = proc.get('id', 'unknown')
            port = proc.get('port', '')
            flags = proc.get('flags', '')
            cmd = proc.get('cmd', '')

            # Truncate command to fit
            if len(cmd) > 61:
                cmd_display = cmd[:58] + "..."
            else:
                cmd_display = cmd

            # Show host only on first row for that host
            show_host = host if host != prev_host else ''

            lines.append(f"│ {show_host[:19].ljust(19)}│ {proc_id[:19].ljust(19)}│ {str(port)[:6].ljust(6)}│ {flags[:5].ljust(5)}│ {cmd_display[:61].ljust(61)}│")

            prev_host = host

        lines.append("└" + "─" * 20 + "┴" + "─" * 20 + "┴" + "─" * 7 + "┴" + "─" * 6 + "┴" + "─" * 62 + "┘")

        return "\n".join(lines)

    def _format_lane_section(self, lane_usage: List[Tuple[str, str, str, str]]) -> str:
        """Format the lane usage section"""
        lines = []
        lines.append("┌" + "─" * 78 + "┐")
        lines.append("│ " + "Lane Usage by Host (DRP nodes only)".ljust(77) + "│")
        lines.append("├" + "─" * 20 + "┬" + "─" * 12 + "┬" + "─" * 8 + "┬" + "─" * 36 + "┤")
        lines.append("│ " + "Host".ljust(19) + "│ " + "Device".ljust(11) + "│ " + "Lane".ljust(7) + "│ " + "Process ID".ljust(35) + "│")
        lines.append("├" + "─" * 20 + "┼" + "─" * 12 + "┼" + "─" * 8 + "┼" + "─" * 36 + "┤")

        prev_host = None
        prev_device = None
        for host, device, lane, proc_id in lane_usage:
            # Show host only on first row for that host
            show_host = host if host != prev_host else ''
            # Show device only on first row for that host+device combo
            show_device = device if (host != prev_host or device != prev_device) else ''

            lines.append(f"│ {show_host[:19].ljust(19)}│ {show_device[:11].ljust(11)}│ {lane[:7].ljust(7)}│ {proc_id[:35].ljust(35)}│")

            prev_host = host
            prev_device = device

        lines.append("└" + "─" * 20 + "┴" + "─" * 12 + "┴" + "─" * 8 + "┴" + "─" * 36 + "┘")

        return "\n".join(lines)

    def _format_process_types(self) -> str:
        """Format the process types section"""
        lines = []
        lines.append("┌" + "─" * 78 + "┐")
        lines.append("│ " + "Process Types".ljust(77) + "│")
        lines.append("├" + "─" * 25 + "┬" + "─" * 7 + "┬" + "─" * 45 + "┤")
        lines.append("│ " + "Type".ljust(24) + "│ " + "Count".ljust(6) + "│ " + "Process IDs".ljust(44) + "│")
        lines.append("├" + "─" * 25 + "┼" + "─" * 7 + "┼" + "─" * 45 + "┤")

        process_types = self.analyzer.get_process_types()

        for ptype in sorted(process_types.keys()):
            procs = process_types[ptype]
            count = len(procs)

            proc_list = ', '.join(procs)

            # Word wrap the process list if it's too long
            first_line = True
            remaining = proc_list
            while remaining:
                if len(remaining) <= 44:
                    # Fits on one line
                    if first_line:
                        lines.append(f"│ {ptype[:24].ljust(24)}│ {str(count).rjust(5)} │ {remaining.ljust(44)}│")
                        first_line = False
                    else:
                        lines.append(f"│ {''.ljust(24)}│ {''.ljust(6)}│ {remaining.ljust(44)}│")
                    break
                else:
                    # Need to wrap - find last comma before 44 chars
                    wrap_pos = remaining[:44].rfind(',')
                    if wrap_pos == -1:
                        # No comma found, just split at 44
                        wrap_pos = 44
                    else:
                        wrap_pos += 1  # Include the comma

                    chunk = remaining[:wrap_pos].rstrip(', ')
                    if first_line:
                        lines.append(f"│ {ptype[:24].ljust(24)}│ {str(count).rjust(5)} │ {chunk.ljust(44)}│")
                        first_line = False
                    else:
                        lines.append(f"│ {''.ljust(24)}│ {''.ljust(6)}│ {chunk.ljust(44)}│")

                    # Continue with remainder
                    remaining = remaining[wrap_pos:].lstrip(', ')

        lines.append("└" + "─" * 25 + "┴" + "─" * 7 + "┴" + "─" * 45 + "┘")

        return "\n".join(lines)

    def _format_transformations(self) -> str:
        """Format the transformations section (renames and extensions)"""
        lines = []
        lines.append("┌" + "─" * 78 + "┐")
        lines.append("│ " + "Config Transformations".ljust(77) + "│")
        lines.append("├" + "─" * 78 + "┤")

        # Renames section
        if self.analyzer.renames:
            lines.append("│ " + "Renamed:".ljust(77) + "│")
            for new_id, old_id in sorted(self.analyzer.renames.items()):
                rename_str = f"  {old_id} → {new_id}"
                lines.append(f"│ {rename_str[:77].ljust(77)}│")

        # Extensions section
        if self.analyzer.extended:
            if self.analyzer.renames:
                lines.append("│ " + "".ljust(77) + "│")  # Blank line separator
            lines.append("│ " + "Extended (added):".ljust(77) + "│")
            extended_list = ', '.join(sorted(self.analyzer.extended))
            # Word wrap if needed
            remaining = f"  {extended_list}"
            while remaining:
                if len(remaining) <= 77:
                    lines.append(f"│ {remaining.ljust(77)}│")
                    break
                else:
                    wrap_pos = remaining[:77].rfind(',')
                    if wrap_pos == -1:
                        wrap_pos = 77
                    else:
                        wrap_pos += 1
                    lines.append(f"│ {remaining[:wrap_pos].ljust(77)}│")
                    remaining = "  " + remaining[wrap_pos:].lstrip(', ')

        lines.append("└" + "─" * 78 + "┘")

        return "\n".join(lines)

    def _format_quick_stats(self) -> str:
        """Format quick statistics"""
        hosts = self.analyzer.get_hosts()
        used_nodes = len([h for h in hosts.keys() if not h.startswith('localhost')])

        lines = []
        lines.append("Quick Stats:")
        lines.append(f"  • Nodes in use: {used_nodes}")
        lines.append(f"  • Total processes: {len(self.analyzer.processes)}")

        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Parse and analyze LCLS2 DAQ CNF configuration files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s hsd.cnf
  %(prog)s tmo.py rix.py
  %(prog)s tmo_sc.py --verbose
        """
    )

    parser.add_argument('files', nargs='+', metavar='FILE',
                       help='CNF or Python configuration file(s) to parse')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show verbose output')
    parser.add_argument('--no-imports', action='store_true',
                       help='Do not follow import chains')

    args = parser.parse_args()

    try:
        # Parse all files
        parsers = []
        for file in args.files:
            try:
                cnf_parser = CnfParser(file, follow_imports=not args.no_imports)
                parsers.append(cnf_parser)
            except Exception as e:
                print(f"Error parsing {file}: {e}", file=sys.stderr)
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                return 1

        if not parsers:
            print("No files successfully parsed", file=sys.stderr)
            return 1

        # Analyze resources
        analyzer = ResourceAnalyzer(parsers)

        # Format and display output
        formatter = OutputFormatter(parsers, analyzer, verbose=args.verbose)
        print(formatter.format_summary())

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())