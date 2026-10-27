"""Discover, select, acquire and import reviewed public development audio."""
import argparse
import json
from pathlib import Path

from src.lava.jaes_v2.acquisition import discover, acquire_selection
from src.lava.jaes_v2.evidence import write_json, write_csv
from src.lava.jaes_v2.mlaad import metadata_inventory, select_development, import_mlaad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('discover', 'select', 'acquire', 'import'))
    parser.add_argument('--discovery', type=Path, required=True)
    parser.add_argument('--revision')
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--per-group', type=int, default=24)
    parser.add_argument('--group-count', type=int, default=4)
    parser.add_argument('--language', default='en')
    parser.add_argument('--seed', type=int, default=3804)
    parser.add_argument('--groups', nargs='+')
    parser.add_argument('--sample-ids', type=Path)
    parser.add_argument('--maximum-bytes', type=int, default=150_000_000)
    parser.add_argument('--purpose', choices=('noncommercial_research',))
    args = parser.parse_args()
    if args.mode == 'discover':
        result = discover(args.discovery, revision=args.revision)
        print({key:result[key] for key in ('status','revision','repository_audio_files')})
    elif args.mode == 'select':
        if not args.selection or args.selection.exists():
            parser.error('A new --selection path is required')
        records, summary = metadata_inventory(args.discovery)
        selected_ids = set(args.sample_ids.read_text().splitlines()) if args.sample_ids else None
        selected = select_development(records, revision=summary['revision'], per_group=args.per_group,
                                      group_count=args.group_count, language=args.language, seed=args.seed,
                                      selected_groups=args.groups, selected_ids=selected_ids)
        write_json(args.selection, selected)
        write_json(args.selection.with_suffix('.inventory.json'), summary)
        write_csv(args.discovery / (args.selection.stem + '.inventory.csv'), records)
        print({**summary, 'selected':len(selected['records']), 'groups':selected['groups'],
               'expected_audio_bytes':sum(row['expected_size'] for row in selected['records'])})
    else:
        if not args.selection or not args.destination:
            parser.error('--selection and --destination are required')
        selection = json.loads(args.selection.read_text())
        if args.mode == 'acquire':
            if not args.purpose:
                parser.error('--purpose noncommercial_research is required')
            print(acquire_selection(args.discovery, selection, args.destination,
                                    maximum_bytes=args.maximum_bytes, purpose=args.purpose)['status'])
        else:
            if not args.output:
                parser.error('A new --output import directory is required')
            print(import_mlaad(args.discovery, args.destination, selection, args.output))


if __name__ == '__main__':
    main()
