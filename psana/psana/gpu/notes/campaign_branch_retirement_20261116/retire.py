"""Guarded, one-time retirement of three preserved campaign source branches."""
import argparse
import datetime
import json
from pathlib import Path
import re
import subprocess


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run(args):
    return subprocess.check_output(args, text=True).strip()


def git(repo, *args):
    return run(['git', '-C', str(repo), *args])


def api(path):
    return json.loads(run(['gh', 'api', path]))


def remote_refs(repo, remote, *refs):
    output = git(repo, 'ls-remote', remote, *refs)
    return {line.split()[1]: line.split()[0] for line in output.splitlines()}


def local_refs(repo, plan):
    rows = git(repo, 'for-each-ref', '--format=%(refname) %(objectname)',
               'refs/heads', 'refs/remotes').splitlines()
    result = {}
    for row in rows:
        ref, sha = row.split()
        for source in plan['sources']:
            if ref == 'refs/heads/' + source['branch'] or ref.endswith('/' + source['branch']):
                require(sha == source['commit'], 'Unexpected local source ref: ' + ref)
                result[ref] = sha
    return result


def check_worktree(entry):
    path = Path(entry['path'])
    require(path.is_dir() and not path.is_symlink(), 'Missing/symlinked source worktree')
    require(git(path, 'rev-parse', 'HEAD') == entry['commit'], 'Worktree HEAD changed')
    require(git(path, 'symbolic-ref', 'HEAD') == 'refs/heads/' + entry['branch'],
            'Worktree branch changed')
    require(not git(path, 'status', '--porcelain', '--untracked-files=all'),
            'Source worktree has tracked or untracked changes')
    require(not git(path, 'ls-files', '--others', '--ignored', '--exclude-standard'),
            'Source worktree has ignored files')


def check_prs(plan):
    repositories = sorted({source['repository'] for source in plan['sources']})
    checked = {}
    for repo in repositories:
        count, page = 0, 1
        while True:
            pulls = api(f'repos/{repo}/pulls?state=open&per_page=100&page={page}')
            require(isinstance(pulls, list), 'Invalid PR listing')
            count += len(pulls)
            for pr in pulls:
                for source in plan['sources']:
                    for side in ('head', 'base'):
                        owner = (pr[side].get('repo') or {}).get('full_name')
                        require(not (owner == source['repository'] and
                                     pr[side]['ref'] == source['branch']),
                                f'Open PR dependency: {repo}#{pr["number"]} {side}')
            if len(pulls) < 100:
                break
            page += 1
        checked[repo] = count
    for source in plan['sources']:
        if source['closed_pr']:
            pr = api(f'repos/{source["repository"]}/pulls/{source["closed_pr"]}')
            require(pr['state'] == 'closed' and not pr['merged'], 'Archive PR state changed')
            require(pr['head']['sha'] == source['commit'], 'Archive PR head changed')
    return checked


def preflight(plan):
    repo = Path(plan['canonical_worktree'])
    require(git(repo, 'branch', '--show-current') == plan['canonical_branch'],
            'Not the canonical results branch')
    canonical = 'refs/heads/' + plan['canonical_branch']
    require(remote_refs(repo, 'origin', canonical).get(canonical) ==
            plan['published_results_commit'], 'Published results branch changed')
    for source in plan['sources']:
        expected_url = 'git@github.com:' + source['repository'] + '.git'
        require(git(repo, 'remote', 'get-url', source['remote']) == expected_url,
                'Remote fetch URL changed')
        require(git(repo, 'remote', 'get-url', '--push', source['remote']) == expected_url,
                'Remote push URL changed')
        branch = 'refs/heads/' + source['branch']
        require(remote_refs(repo, source['remote'], branch).get(branch) == source['commit'],
                'Source branch moved or disappeared: ' + branch)
        require(git(repo, 'cat-file', '-t', source['commit']) == 'commit', 'Source object missing')
        tag = 'refs/tags/' + source['tag']
        require(not git(repo, 'tag', '--list', source['tag']), 'Archive tag already exists locally')
        require(not remote_refs(repo, source['remote'], tag, tag + '^{}'),
                'Archive tag already exists remotely; inspect before resuming')
    for entry in plan['remove_worktrees']:
        check_worktree(entry)
        require(Path(entry['path']).resolve() != repo.resolve(), 'Cannot remove active worktree')
    refs = {path: local_refs(path, plan) for path in plan['local_repositories']}
    return dict(status='pass', local_refs=refs, open_prs_inspected=check_prs(plan),
                scope='Known Perlmutter object stores and both GitHub source repositories')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--execute', action='store_true')
    parser.add_argument('--approval', type=Path)
    args = parser.parse_args()
    plan = json.loads(Path(__file__).with_name('plan.json').read_text())
    repo = Path(plan['canonical_worktree'])
    if args.preflight:
        print(json.dumps(preflight(plan), indent=2))
        return
    require(args.approval is not None, 'Independent review receipt required')
    approval = json.loads(args.approval.read_text())
    require(approval['verdict'] == 'pass' and
            not any(f['blocking'] for f in approval['findings']), 'Review did not pass')
    require(git(repo, 'rev-parse', 'HEAD') == approval['reviewed_commit'],
            'Execution checkout differs from reviewed commit')
    require(not git(repo, 'status', '--porcelain', '--untracked-files=all'),
            'Execution worktree is not clean')
    receipt_path = Path(plan['execution_receipt'])
    require(not receipt_path.exists(), 'Prior execution exists; inspect before retrying')
    receipt = dict(reviewed_commit=approval['reviewed_commit'], preflight=preflight(plan), steps=[])

    def record(step, **details):
        receipt['steps'].append(dict(step=step, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **details))
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt['steps'][-1]), flush=True)

    record('preflight passed')
    for source in plan['sources']:
        tag = 'refs/tags/' + source['tag']
        message = (f'Archive {source["repository"]}:{source["branch"]} at {source["commit"]}.\n'
                   'Campaign documents consolidated on slac-lcls/lcls2 results/psana2-gpu-scaling.\n'
                   'Preserve original source history before retiring the obsolete branch.\n')
        git(repo, 'tag', '-a', source['tag'], source['commit'], '-m', message)
        record('local annotated tag created', tag=source['tag'], commit=source['commit'])
        git(repo, 'push', source['remote'], tag + ':' + tag)
        actual = remote_refs(repo, source['remote'], tag, tag + '^{}')
        require(actual.get(tag) == git(repo, 'rev-parse', tag) and
                actual.get(tag + '^{}') == source['commit'], 'Remote archive tag verification failed')
        record('remote annotated tag verified', repository=source['repository'], tag=source['tag'],
               tag_object=actual[tag], commit=actual[tag + '^{}'])
    record('all archive tags verified; deletion gate satisfied')
    record('fresh PR dependencies checked', open_prs_inspected=check_prs(plan))
    for source in plan['sources']:
        ref = 'refs/heads/' + source['branch']
        require(remote_refs(repo, source['remote'], ref).get(ref) == source['commit'],
                'Source advanced after archival; preserve branch')
        git(repo, 'push', '--force-with-lease=' + ref + ':' + source['commit'],
            source['remote'], ':' + ref)
        require(not remote_refs(repo, source['remote'], ref), 'Remote branch still present')
        record('remote branch deleted', repository=source['repository'], branch=source['branch'],
               expected_commit=source['commit'])
    for entry in plan['remove_worktrees']:
        check_worktree(entry)
        git(repo, 'worktree', 'remove', entry['path'])
        require(not Path(entry['path']).exists(), 'Source worktree still exists')
        record('clean obsolete worktree removed', path=entry['path'])
    for path in plan['local_repositories']:
        refs = local_refs(path, plan)
        registered = git(path, 'worktree', 'list', '--porcelain')
        for ref, sha in refs.items():
            require('branch ' + ref + '\n' not in registered + '\n', 'Ref still checked out')
            git(path, 'update-ref', '-d', ref, sha)
            if ref.startswith('refs/heads/'):
                branch = ref[len('refs/heads/'):]
                pattern = '^branch\\.' + re.escape(branch) + '\\.'
                settings = subprocess.run(['git', '-C', path, 'config', '--get-regexp', pattern],
                                          capture_output=True, text=True)
                require(settings.returncode in (0, 1), 'Branch configuration read failed')
                if settings.returncode == 0:
                    git(path, 'config', '--remove-section', 'branch.' + branch)
            record('local obsolete ref deleted', repository=path, ref=ref, expected_commit=sha)
        require(not local_refs(path, plan), 'Obsolete local refs remain')
    for source in plan['sources']:
        tag = 'refs/tags/' + source['tag']
        actual = remote_refs(repo, source['remote'], tag, tag + '^{}', 'refs/heads/' + source['branch'])
        require(actual.get(tag + '^{}') == source['commit'] and
                actual.get(tag) == git(repo, 'rev-parse', tag) and
                'refs/heads/' + source['branch'] not in actual, 'Final remote verification failed')
    require(git(repo, 'branch', '--show-current') == plan['canonical_branch'], 'Active branch changed')
    record('complete', remote_branches_removed=3, obsolete_worktrees_removed=1,
           archived_source_commits=3, canonical_branch=plan['canonical_branch'])


if __name__ == '__main__':
    main()
