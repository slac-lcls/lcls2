Review eafec66243fa387753eab37f3f8097ca2a41d860 against base e118b3c7eec28203f06a1c9594306d59b58af4c5. This is a documentation/audit-helper correction, NOT a Stage 3 writer implementation. A prior attempt timed out without any verdict after broad source exploration; no approval or findings resulted. Please keep this review focused and conclude within this invocation. Report blocked if evidence is insufficient, rather than expanding into the entire dataset/writer implementation.

Read these four bounded inputs:
1. notes/synthetic_mfx_stage1_20261103/review_20261104/responses.md — supplied findings C1–C7 and dispositions.
2. .../review_20261104/audit_stage3_inputs.py — the new read-only SMD prerequisite checker.
3. psana/psana/gpu/docs/proposals/persistent_task_state_xpp_cube.md, ONLY the synthetic dataset section starting at line 559.
4. .../review_20261104/access_and_git.txt — current account, ACL and Git evidence.
The scope is to check whether this response correctly specifies scan schema, all-stream runinfo, complete transition headers, PV provenance/run-2 pinning, remaining collaborator access, validation environment/discovery, and the existing Git checkpoint. Identify contradictions or false completion claims. Keep incoming IDs C1–C7 if reopened; use stable new IDs otherwise. At most a few targeted extra reads if essential, then conclude. Do not investigate unrelated GPU proposal text or implement the writer mentally.

The new helper executed successfully on all ten frozen SMD files. Their hashes matched manifest.json; no scan detector, 24-byte BeginSteps, identical source run identity and no Enable chunkinfo payload; 70 templates recorded in source_preflight.json. This is supplied coordinator evidence, not a test run by you. Syntax/link checks, git diff --check and all applicable repository precommit checks passed. All ten original Stage 1/2 note files remain unchanged. C7's response accidentally says eleven note files; the evidence lists ten plus the plan, and the evidence-only follow-up corrects that count. No implementation requirement changes are pending.

Actual collaborator-session readback is explicitly pending and belongs to later acceptance; do not count mere membership as a completed read. Stage 3 generation and Stage 4 DataSource execution are explicitly unexecuted. No new DB or logbook calls are claimed. See supplied direct source excerpts below, all read from the unchanged checked-out source. No need to explore unrelated source files to verify these narrow contracts.

SOURCE psdaq/psdaq/control/BlueskyScan.py lines 102–138:
102:                 my_data = {}
103:
104:                 # record step_value
105:                 my_data.update({'step_value': self.step_value})
106:
107:                 # record motor positions.  BeginStep scan fields default to
108:                 # scanvarN by motor order; motor.name is recorded only as
109:                 # metadata in step_docstring unless scan_names overrides it.
110:                 scan_doc = {
111:                     "detname": "scan",
112:                     "scantype": self.scantype,
113:                     "step": self.step_value,
114:                 }
115:                 scan_motors = [
116:                     motor for motor in self.motors
117:                     if motor.name != ControlDef.STEP_VALUE
118:                 ]
119:                 for motor_index, motor in enumerate(scan_motors):
120:                     record_name = self._scan_record_name(motor, motor_index)
121:                     my_data.update({record_name: motor.position})
122:                     scan_doc[record_name] = motor.name
123:
124:                 # record step_docstring
125:                 my_data.update({'step_docstring': json.dumps(scan_doc)})
126:
127:                 data = {
128:                     "motors":           my_data,
129:                     "timestamp":        0,
130:                     "detname":          "scan",
131:                     "dettype":          "scan",
132:                     "scantype":         self.scantype,
133:                     "serial_number":    "1234",
134:                     "alg_name":         "raw",
135:                     "alg_version":      [1,0,0],
136:                 }
137:
138:                 configureBlock = self.getBlock(transition="Configure", data=data)

SOURCE psana/psana/psexp/ds_base.py lines 973–985:
973:     def _get_runinfo(self):
974:         expt, runnum, timestamp = (None, None, None)
975:
976:         if self.beginruns:
977:             beginrun_dgram = self.beginruns[0]
978:             if hasattr(beginrun_dgram, "runinfo"):  # some xtc2 do not have BeginRun
979:                 # BeginRun in all streams have the same information.
980:                 # We just need to get one from the first segment/ dgram.
981:                 segment_id = list(beginrun_dgram.runinfo.keys())[0]
982:                 expt = beginrun_dgram.runinfo[segment_id].runinfo.expt
983:                 runnum = beginrun_dgram.runinfo[segment_id].runinfo.runnum
984:                 timestamp = beginrun_dgram.timestamp()
985:         return expt, runnum, timestamp

SOURCE psana/psana/detector/envstore.py lines 6–28:
6:
7: class EnvImpl(DetectorImpl):
8:     def __init__(self, *args):
9:         super(EnvImpl, self).__init__(*args)
10:
11:     def __call__(self, events):
12:         if self._env_store is None:
13:             err_msg = f"Function call is not available for this detector."
14:             raise MissingEnvStore(err_msg)
15:
16:         if isinstance(events, list):
17:             return self._env_store.values(events, self._var_name)
18:         elif isinstance(events, Event):
19:             env_values = self._env_store.values([events], self._var_name)
20:             return env_values[0]
21:         elif isinstance(events, Step):
22:             env_values = self._env_store.values([events.evt], self._var_name)
23:             return env_values[0]
24:         err_msg = f"Calling detector only accept Event or Step. Invalid type: {type(events)} given."
25:         raise InvalidInputEnvStore(err_msg)
26:
27:     @property
28:     def dtype(self):

SOURCE psana/psana/detector/misc_detectors.py lines 52–56:
52: class pv_raw_1_0_0(DetectorImpl):
53:     def __init__(self, *args):
54:         super().__init__(*args)
55:         self._add_fields()
56:

SOURCE psana/psana/smdreader.pyx lines 155–161:
155:         # Index of the slowest detector or intg_stream_id if given.
156:         self.winner             = -1
157:
158:         # Sets event frequency that fake EndStep/BeginStep pair is inserted.
159:         self.fakestep_flag = int(os.environ.get('PS_FAKESTEP_FLAG', 0))
160:
161:     cdef void _init_send_buffers(self):

SOURCE xtcdata/xtcdata/xtc/NamesId.hh lines 23–51:
23:     // must be consistent with the total number of bits used below.
24:     // this will determine the size of the Names lookup array,
25:     // so we try not to make it too large
26:     enum {NumberOf=1<<20};
27:
28:     NamesId() : Src() {}  // undefined Src
29:
30:     NamesId(unsigned nodeId, unsigned namesId) : Src(((nodeId&0xfff)<<8)|(namesId&0xff)) {
31:         if ((nodeId&0xfff) != nodeId) {
32:             printf("*** %s:%d: nodeId too large %d\n",__FILE__,__LINE__,nodeId);
33:             abort();
34:         }
35:         if ((namesId&0xff) != namesId) {
36:             printf("*** %s:%d: namesId too large %d\n",__FILE__,__LINE__,namesId);
37:             abort();
38:         }
39:     }
40:
41:     operator unsigned() const {return value();}
42:
43:     unsigned nodeId()  {return (_value>>8)&0xfff;}
44:     unsigned namesId() {return _value&0xff;}
45:
46: };
47:
48: }
49: #endif

SOURCE xtcdata/xtcdata/xtc/Dgram.hh lines 15–51:
15: public:
16:     enum Type { Event = 0, Occurrence = 1, Marker = 2 };
17:     enum { NumberOfTypes = 3 };
18:     TransitionBase() {}
19:     TransitionBase(Type type_, TransitionId::Value tid_,
20:                    const TimeStamp& time_, uint32_t env_) :
21:         time(time_), env((type_<<28)|(tid_<<24)|(env_&0xffffff)) {}
22: public:
23:     uint16_t readoutGroups() const { return (env)&0xffff; }
24: public:
25:     TimeStamp time;
26:     uint32_t env;
27: };
28:
29: class Transition : public TransitionBase {
30: public:
31:     Transition() {}
32:     Transition(Type type_, TransitionId::Value tid_,
33:                const TimeStamp& time_, uint32_t env_) :
34:       TransitionBase(type_, tid_, time_, env_) {}
35: public:
36:     unsigned control()            const { return (env>>24)&0xff; }
37:     TransitionId::Value service() const { return TransitionId::Value(control()&0xf); }
38:     Type type()                   const { return Type((control()>>4)&0x3); }
39:     bool isEvent()                const { return service()==TransitionId::L1Accept; }
40:     bool isExtended()             const { return ((env>>16)&1) && !isEvent(); }
41: };
42:
43: class Dgram : public Transition {
44: public:
45:     static const unsigned MaxSize = 0x1000000;
46:     Dgram() {}
47:     Dgram(const Transition& transition_) :
48:         Transition(transition_) { }
49:     Dgram(const Transition& transition_, const Xtc& xtc_) :
50:         Transition(transition_), xtc(xtc_)  { }
51: public:

SOURCE xtcdata/xtcdata/xtc/Xtc.hh lines 134–142:


Use only file reads. Do not run commands, mutate state, delegate or contact services. Produce the required structured verdict; do not claim fresh tests. status.json is a pre-review checkpoint; actual review verdict and SHA will be saved afterward.
