"""Fresh final coverage fixtures for the lower-field coset experiment.

These topics have not appeared in the preceding scalar, learned/E8 or stacked
experiments. Capture only after the new codec/configuration freeze. Fixtures
remain original synthetic coverage, not a representative benchmark sample.
Length variants share numbered filler and are correlated observations.
"""


FINAL = [
    dict(
        name="coset_puppet_rehearsal",
        kind="text",
        prefix="A community puppet theatre rehearses its opening performance.\n",
        body=(
            "Rehearsal <cycle>: The puppeteer adjusted the strings while the "
            "stage manager checked the painted scenery. A small bell marked "
            "each change of scene. Behind the curtain, the musicians practiced "
            "the entrance melody and waited for the lantern to dim. The "
            "director asked everyone to leave enough silence for the audience "
            "to notice the wooden bird above the doorway. "
        ),
        suffix=(
            "\nWhen the final rehearsal ended, the stage manager decided to"
        ),
    ),
    dict(
        name="coset_javascript_run_length",
        kind="code",
        prefix="JavaScript implementation of run-length encoding.\n",
        body=(
            "// example <cycle>\n"
            "function encodeRuns(input) {\n"
            "  const runs = [];\n"
            "  for (const symbol of input) {\n"
            "    const last = runs[runs.length - 1];\n"
            "    if (last && last[0] === symbol) {\n"
            "      last[1] += 1;\n"
            "    } else {\n"
            "      runs.push([symbol, 1]);\n"
            "    }\n"
            "  }\n"
            "  return runs;\n"
            "}\n"
            "// Empty input produces no runs. Adjacent equal symbols merge.\n"
        ),
        suffix=(
            "\n// Explain why the condition checks last before reading last[0].\n"
        ),
    ),
    dict(
        name="coset_chess_league",
        kind="structured",
        prefix="Community chess league score ledger.\n",
        body=(
            "Match <cycle>: division=junior; table=11; white=Farah; "
            "black=Emil; result=draw; time_control=rapid; arbiter=Noor. "
            "Both players sign the score sheet before the result is posted.\n"
        ),
        suffix=(
            "\nFINAL RESULT: match=TY-583; table=26; white=Selin; "
            "black=Mateo; result=white_win; arbiter=OWLET-64.\n"
            "Question: What is the arbiter identifier for match TY-583?\n"
            "Answer:"
        ),
    ),
]


def fixture_manifest():
    return {
        "namespace": "kv_coset_fields",
        "provenance": "Original synthetic fixtures, new to this experiment",
        "fixture_names": [row["name"] for row in FINAL],
        "kinds": [row["kind"] for row in FINAL],
        "lengths": [512, 2048],
        "caveat": (
            "Three fresh topics provide six correlated prompt/length coverage "
            "cases. They are not a representative workload or task-quality "
            "benchmark; model generation checks compare identical execution."
        ),
    }
