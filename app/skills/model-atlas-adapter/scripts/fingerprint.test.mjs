import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile, unlink, symlink } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { snapshot, verify } from './fingerprint.mjs';

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), 'model-atlas-fingerprint-'));
  await mkdir(join(root, 'app'));
  await writeFile(join(root, 'app', 'a.ts'), 'const n = 1;\n');
  await writeFile(join(root, 'config.json'), '{}\n');
  return root;
}

test('stable hashes use normalized, sorted scopes and include Unicode filenames', async () => {
  const root = await fixture();
  await mkdir(join(root, 'app', 'nested'));
  await writeFile(join(root, 'app', 'nested', '符号.ts'), 'D_idx');
  const first = await snapshot(root, ['config.json', 'app\\nested', 'app', 'config.json']);
  const second = await snapshot(root, ['app', 'config.json', 'app/nested']);
  assert.deepEqual(first, second);
  assert.equal(first.files.length, 3);
  assert.equal((await verify(root, first)).valid, true);
});

test('detects newly added untracked files, changed bytes, and deletions', async () => {
  const root = await fixture();
  const baseline = await snapshot(root, ['app', 'config.json']);
  await writeFile(join(root, 'app', 'new.ts'), 'new untracked source');
  assert.deepEqual((await verify(root, baseline)).changes.map(x => x.path), ['app/new.ts']);
  await unlink(join(root, 'app', 'new.ts'));
  await writeFile(join(root, 'app', 'a.ts'), 'const n = 2;\n');
  assert.deepEqual((await verify(root, baseline)).changes.map(x => x.path), ['app/a.ts']);
  await unlink(join(root, 'app', 'a.ts'));
  assert.equal((await verify(root, baseline)).valid, false);
});

test('missing explicit files are recorded and newly created files invalidate them', async () => {
  const root = await fixture();
  const baseline = await snapshot(root, ['optional.json']);
  assert.deepEqual(baseline.files, [{ path: 'optional.json', sha256: null }]);
  await writeFile(join(root, 'optional.json'), '{}');
  assert.equal((await verify(root, baseline)).valid, false);
});

test('rejects corrupted manifests and traversal/root-wide scopes', async () => {
  const root = await fixture();
  const baseline = await snapshot(root, ['app']);
  baseline.files[0].sha256 = 'tampered';
  await assert.rejects(verify(root, baseline), /corrupted/);
  for (const scope of ['.', '..', '../outside', 'app/../config.json', '/root', 'C:\\outside']) {
    await assert.rejects(snapshot(root, [scope]), /root-relative/);
  }
});

test('rejects symlinked ancestors as well as direct links', async t => {
  const root = await fixture();
  const other = await fixture();
  try {
    await symlink(other, join(root, 'linked'), 'junction');
  } catch (error) {
    if (['EPERM', 'EACCES', 'ENOSYS'].includes(error.code)) return t.skip(error.code);
    throw error;
  }
  await assert.rejects(snapshot(root, ['linked/app']), /Symlink/);
});

test('CLI emits JSON and exits nonzero for stale evidence', async () => {
  const root = await fixture();
  const script = fileURLToPath(new URL('./fingerprint.mjs', import.meta.url));
  const run = (...args) => spawnSync(process.execPath, [script, ...args], { encoding: 'utf8', windowsHide: true });
  const initial = run('snapshot', root, 'app');
  assert.equal(initial.status, 0, initial.stderr);
  const manifest = join(root, 'manifest.json');
  await writeFile(manifest, initial.stdout);
  assert.equal(run('verify', root, manifest).status, 0);
  await writeFile(join(root, 'app', 'a.ts'), 'changed');
  const stale = run('verify', root, manifest);
  assert.equal(stale.status, 1);
  assert.equal(JSON.parse(stale.stdout).valid, false);
});
