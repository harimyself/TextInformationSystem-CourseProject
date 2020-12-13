import spacy
import os
import json
import sys
import errno

"""
    Script to extract names and organizations on the faculty bios pages and save them in
    given directories for later use during indexing

    We use spacy library here
"""

ner_model = spacy.load('en_core_web_lg')

def save_names_and_organizations(bios_dir, bios_file_name, names_file_directory, org_file_directory):
    page_text = ''
    with open(os.path.join(bios_dir, bios_file_name), 'r') as fh:
        page_text = ' '.join([line for line in fh])
    
    ner_data = ner_model(page_text)
    name = ''
    for ent in ner_data.ents:
        if ent.label_ == 'PERSON':
            name = ent.text
            break

    orgs = []
    for ent in ner_data.ents:
        if ent.label_ == 'ORG':
            orgs.append(ent.text)

    with open(os.path.join(names_file_directory, bios_file_name), 'w') as fh:
        fh.write(name)
    
    with open(os.path.join(org_file_directory, bios_file_name), 'w') as fh:
        json.dump(orgs, fh)


def run_extraction(bios_dir, names_dir, org_dir):
    counter = 1
    total_pages = len(os.listdir(bios_dir))
    for fname in os.listdir(bios_dir):
        save_names_and_organizations(bios_dir, fname, names_dir, org_dir)
        if counter % 100 == 0:
            print(f"Done with {counter}/{total_pages} faculty_pages")
        counter += 1

def main():
    bios_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__)) + '/../data/compiled_bios/'
    names_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(__file__)) + '/../data/compiled_bios_names/'
    org_dir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(os.path.abspath(__file__)) + '/../data/compiled_bios_orgs/'

    if not os.path.exists(names_dir):
        try:
            os.makedirs(names_dir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    
    if not os.path.exists(org_dir):
        try:
            os.makedirs(org_dir)
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
                
    run_extraction(bios_dir, names_dir, org_dir)

if __name__ == "__main__":
    main()
