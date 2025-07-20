#!/bin/bash

# Slurmd Start Script for daq-det-evr01
# Usage: ./slurmd_start.sh [start|stop|restart|status]

# Configuration
SLURM_HOME="/cds/home/m/monarin/slurmd_bundle"
MUNGE_HOME="/cds/home/m/monarin/munge_tools"
SLURM_CONF="${SLURM_HOME}/etc/slurm.conf"
SLURMD_BIN="${SLURM_HOME}/bin/slurmd"
SLURMSTEPD_BIN="${SLURM_HOME}/sbin/slurmstepd"
SLURM_LOG_DIR="/cds/home/m/monarin/slurm_tmp/log"
PIDFILE="/cds/home/m/monarin/slurm_tmp/slurmd.pid"

# Export library paths
export LD_LIBRARY_PATH="${SLURM_HOME}/lib:${MUNGE_HOME}/lib:${LD_LIBRARY_PATH}"

# Function to check if slurmd is running
is_running() {
    if [ -f "$PIDFILE" ]; then
        local pid=$(cat "$PIDFILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PIDFILE"
            return 1
        fi
    fi
    return 1
}

# Function to start slurmd
start_slurmd() {
    if is_running; then
        echo "Slurmd is already running (PID: $(cat $PIDFILE))"
        return 1
    fi

    echo "Starting slurmd..."

    # Create log directory if it doesn't exist
    mkdir -p "$SLURM_LOG_DIR"

    # Start slurmd in background
    sudo LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
        "$SLURMD_BIN" \
        -f "$SLURM_CONF" \
        -d "$SLURMSTEPD_BIN" \
        -L "$SLURM_LOG_DIR/slurmd.log" \
        > "$SLURM_LOG_DIR/slurmd.out" 2>&1 &

    local pid=$!
    echo "$pid" > "$PIDFILE"

    # Wait a moment and check if it started successfully
    sleep 2
    if is_running; then
        echo "Slurmd started successfully (PID: $pid)"
        echo "Log file: $SLURM_LOG_DIR/slurmd.log"
        return 0
    else
        echo "Failed to start slurmd"
        rm -f "$PIDFILE"
        echo "Check logs: $SLURM_LOG_DIR/slurmd.out"
        return 1
    fi
}

# Function to stop slurmd
stop_slurmd() {
    if ! is_running; then
        echo "Slurmd is not running"
        return 1
    fi

    local pid=$(cat "$PIDFILE")
    echo "Stopping slurmd (PID: $pid)..."

    sudo kill "$pid"

    # Wait for graceful shutdown
    local count=0
    while is_running && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done

    if is_running; then
        echo "Force killing slurmd..."
        sudo kill -9 "$pid"
        sleep 1
    fi

    rm -f "$PIDFILE"
    echo "Slurmd stopped"
}

# Function to show status
show_status() {
    if is_running; then
        local pid=$(cat "$PIDFILE")
        echo "Slurmd is running (PID: $pid)"

        # Show recent log entries
        if [ -f "$SLURM_LOG_DIR/slurmd.log" ]; then
            echo "Recent log entries:"
            tail -5 "$SLURM_LOG_DIR/slurmd.log"
        fi

        # Check cluster status
        echo ""
        echo "Cluster status:"
        scontrol show node daq-det-evr01 2>/dev/null || echo "Cannot connect to slurmctld"
    else
        echo "Slurmd is not running"
    fi
}

# Function to restart slurmd
restart_slurmd() {
    echo "Restarting slurmd..."
    stop_slurmd
    sleep 2
    start_slurmd
}

# Main script logic
case "${1:-start}" in
    start)
        start_slurmd
        ;;
    stop)
        stop_slurmd
        ;;
    restart)
        restart_slurmd
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status}"
        echo ""
        echo "Commands:"
        echo "  start   - Start slurmd daemon"
        echo "  stop    - Stop slurmd daemon"
        echo "  restart - Restart slurmd daemon"
        echo "  status  - Show slurmd status"
        exit 1
        ;;
esac