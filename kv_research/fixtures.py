"""Original synthetic train/dev/test fixtures; no representative-corpus claim."""
from kv_smoke.prompts import build_input_ids

def fixture(name, kind, prefix, body, suffix):
    return dict(name=name,kind=kind,prefix=prefix,body=body,suffix=suffix)

TRAIN = [
 fixture('train_railway', 'text', 'Manual for restoring an old mountain railway.\n',
  'Entry <cycle>: The maintenance crew inspected the sleepers, measured the rail gauge, and recorded wear at each bend. A drainage channel carried melted snow away from the ballast. Before a locomotive could return, the inspector checked the brakes and signed the repair log. ',
  '\nThe foreman reviewed the inspection notes and'),
 fixture('train_graph_python', 'code', 'Python program for traversing a directed graph.\n',
  '## example <cycle>\ndef reachable(graph, source):\n    pending = [source]\n    visited = set()\n    while pending:\n        vertex = pending.pop()\n        if vertex in visited:\n            continue\n        visited.add(vertex)\n        pending.extend(graph.get(vertex, []))\n    return visited\nassert reachable({0: [1], 1: [2]}, 0) == {0, 1, 2}\n',
  '\n# Explain why the visited set is needed.\n'),
 fixture('train_catalog', 'structured', 'Inventory of agricultural laboratory supplies.\n',
  'Batch <cycle>: sample=fern; cabinet=D12; tray=green; temperature=18; counted=24; inspector=Marin. All vials are labeled with the collection date and source plot.\n',
  '\nSummarize the inventory procedure.\n'),
]
DEV = [fixture('dev_sensor_pipeline','mixed','Design notes for a river sensor pipeline.\n',
 'Window <cycle>: A gauge reports timestamp, water level, and battery voltage. The processor rejects duplicate sequence numbers, stores valid readings, and computes a rolling median.\n'
 'def accept(row, seen):\n    if row["seq"] in seen: return False\n    seen.add(row["seq"])\n    return row["voltage"] > 2.5\n'
 '{"station":"north_bridge","level_cm":138,"voltage":3.1,"status":"normal"}\n',
 '\nDescribe what happens to a repeated sequence number.\n')]
TEST = [
 fixture('test_antarctic_story','text','An Antarctic research station prepares for winter.\n',
  'Journal <cycle>: The meteorologist walked between the instruments and cleared frost from the wind vane. Inside the station, the cook prepared a warm meal while the radio operator checked the satellite link. Everyone helped inventory the fuel before the next storm arrived. ',
  '\nWhen the first snow reached the doorway, the station leader'),
 fixture('test_rust_log_parser','code','Rust code for counting valid log records.\n',
  '// case <cycle>\nfn count_valid(input: &str) -> usize {\n    input.lines().filter(|line| {\n        let mut fields = line.split(",");\n        let id = fields.next().unwrap_or("");\n        let state = fields.next().unwrap_or("");\n        !id.is_empty() && state == "ok"\n    }).count()\n}\nassert_eq!(count_valid("r1,ok\\nr2,bad"), 1);\n',
  '\n// Describe the filter and its behavior for an empty line.\n'),
 fixture('test_museum_lookup','structured','Museum shipment manifest. Preserve the final authorization record.\n',
  'Shipment <cycle>: collection=ceramics; origin=East Gallery; destination=Annex C; crate=blue; handler=Lena; authorization=PENDING. The route includes a climate-controlled holding room.\n',
  '\nFINAL AUTHORIZATION: shipment=ZE-417; destination=West Archive; seal=amber; approval=MAJORCA-82.\nQuestion: What is the approval code for ZE-417?\nAnswer:')
]

SPLITS={'train':TRAIN,'dev':DEV,'test':TEST}
