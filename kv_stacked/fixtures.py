"""Fresh final coverage fixtures, unused by previous experiments."""
FINAL=[
 dict(name='stack_coastal_archive',kind='text',prefix='An archivist restores coastal navigation records.\n',
 body='Page <cycle>: Salt had faded the ink along the edge of the chart. The archivist photographed each sheet, compared the lighthouse bearings, and entered uncertain readings in a separate ledger. Outside, fishing boats returned before the evening tide. ',
 suffix='\nAfter comparing the oldest chart with the new survey, the archivist'),
 dict(name='stack_go_ring_buffer',kind='code',prefix='Go implementation of a bounded ring buffer.\n',
 body='// example <cycle>\ntype Ring struct { data []int; next int; count int }\nfunc (r *Ring) Push(x int) {\n    r.data[r.next] = x\n    r.next = (r.next + 1) % len(r.data)\n    if r.count < len(r.data) { r.count++ }\n}\n// The oldest element is overwritten when the ring is full.\n',
 suffix='\n// Explain the purpose of next and count.\n'),
 dict(name='stack_seed_vault',kind='structured',prefix='Seed vault accession ledger.\n',
 body='Accession <cycle>: species=barley; origin=plateau; shelf=F7; container=steel; humidity=low; curator=Ivo; status=registered. A duplicate packet is stored in the remote vault.\n',
 suffix='\nFINAL RECORD: accession=QR-906; species=millet; shelf=J4; verification=SOLSTICE-31.\nQuestion: What is the verification for QR-906?\nAnswer:')]
