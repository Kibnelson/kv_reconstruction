"""Prespecified fresh coverage fixtures for binary residual reconstruction.

Capture only after development selection and source/configuration freeze.
Repeated numbered filler makes length variants correlated, not independent
workload samples. These prompts are synthetic mechanism coverage.
"""

FINAL = [
    dict(name="binary_clock_repair", kind="text",
         prefix="A conservator repairs a nineteenth-century station clock.\n",
         body=("Workshop entry <cycle>: The conservator laid each brass wheel "
               "on a numbered cloth before examining the worn pivot through "
               "a magnifying lens. The pendulum hung motionless beside the "
               "bench. She compared the escapement with a drawing, cleaned "
               "the bearing, and checked that the replacement spring would "
               "not interfere with the hour strike. Outside, a train passed "
               "the station whose clock would soon keep time again. "),
         suffix="\nBefore reassembling the movement, she recorded"),
    dict(name="binary_kotlin_brackets", kind="code",
         prefix="Kotlin function for validating balanced brackets.\n",
         body=("// example <cycle>\n"
               "fun balanced(text: String): Boolean {\n"
               "    val stack = ArrayDeque<Char>()\n"
               "    val expected = mapOf(')' to '(', ']' to '[', '}' to '{')\n"
               "    for (ch in text) {\n"
               "        if (ch in \"([{\") stack.addLast(ch)\n"
               "        else if (ch in expected) {\n"
               "            if (stack.isEmpty() || stack.removeLast() != expected[ch]) return false\n"
               "        }\n"
               "    }\n"
               "    return stack.isEmpty()\n"
               "}\n"),
         suffix="\n// Explain why an empty stack must be checked before removing its last element.\n"),
    dict(name="binary_tidal_station", kind="structured",
         prefix="Tidal monitoring station maintenance records.\n",
         body=("Inspection <cycle>: station=Estuary-North; sensor=pressure; "
               "reference_cm=182; observed_cm=184; battery_percent=91; "
               "technician=Imani; status=verified. The pressure reading is "
               "compared with a staff gauge before calibration is accepted.\n"),
         suffix=("\nFINAL INSPECTION: station=Jetty-East; sensor=radar; "
                 "technician=Mateo; status=verified; certificate=HERON-806.\n"
                 "Question: What is the certificate for Jetty-East?\nAnswer:")),
]

def fixture_manifest():
    return dict(namespace="kv_binary", provenance="Original fresh synthetic coverage fixtures",
                fixture_names=[x["name"] for x in FINAL], kinds=[x["kind"] for x in FINAL],
                lengths=[512, 2048],
                caveat="Three topics and two correlated length variants, not representative serving workloads.")
