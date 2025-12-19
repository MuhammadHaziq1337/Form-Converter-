import json

old = json.load(open('wew.json'))
new = json.load(open('new.json'))

old_sections = old.get('formStructure', [])
new_sections = new.get('formStructure', [])

print(f"Old sections: {len(old_sections)}")
print(f"New sections: {len(new_sections)}")
print()

print("=== OLD SECTIONS ===")
for s in old_sections:
    fields = s.get('fields', [])
    print(f"  {s['section']}: {s['sectionTitle'][:60]}... ({len(fields)} fields)")

print()
print("=== NEW SECTIONS ===")
for s in new_sections:
    fields = s.get('fields', [])
    print(f"  {s['section']}: {s['sectionTitle'][:60]}... ({len(fields)} fields)")

# Count total fields
old_total = sum(len(s.get('fields', [])) for s in old_sections)
new_total = sum(len(s.get('fields', [])) for s in new_sections)
print()
print(f"Total fields - Old: {old_total}, New: {new_total}, Diff: {old_total - new_total}")
