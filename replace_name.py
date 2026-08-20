import os
import glob

for ext in ('**/*.md', '**/*.py'):
    for f in glob.glob(ext, recursive=True):
        if not os.path.isfile(f) or '.venv' in f:
            continue
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        
        new_content = content.replace('Team Peri Peri Fries', 'Team Peri Peri Fries').replace('team-peri-peri-fries', 'team-peri-peri-fries').replace('Team Peri Peri Fries', 'Team Peri Peri Fries').replace('-', '-')
        
        if new_content != content:
            with open(f, 'w', encoding='utf-8') as file:
                file.write(new_content)
