import { createHash } from 'node:crypto';
import { lstat, readdir, readFile, realpath } from 'node:fs/promises';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import { pathToFileURL } from 'node:url';

const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const order = (a, b) => a < b ? -1 : a > b ? 1 : 0;

function scopePath(value) {
  const name = value.replaceAll('\\', '/').replace(/\/+$/, '');
  if (!name || isAbsolute(name) || /^[A-Za-z]:/.test(name) || name.split('/').some(p => !p || p === '.' || p === '..')) {
    throw new Error(`Expected explicit root-relative file or directory: ${value}`);
  }
  return name;
}

function canonical(manifest) {
  return JSON.stringify({ schema: 1, scopes: manifest.scopes, files: manifest.files });
}

/** Re-enumerate declared scopes on every call; no git-tracked-only shortcut. */
export async function snapshot(rootInput, inputScopes) {
  if (!inputScopes?.length) throw new Error('Declare at least one input scope');
  const root = await realpath(rootInput);
  const scopes = [...new Set(inputScopes.map(scopePath))].sort(order);
  const entries = new Map();
  async function visit(name) {
    const path = resolve(root, name);
    const rel = relative(root, path);
    if (isAbsolute(rel) || rel === '..' || rel.startsWith(`..${sep}`)) throw new Error(`Scope escapes root: ${name}`);
    // Check every component so symlinked ancestors cannot bypass the root check.
    const parts = name.split('/');
    for (let i = 1; i <= parts.length; i++) {
      const component = resolve(root, ...parts.slice(0, i));
      try {
        if ((await lstat(component)).isSymbolicLink()) throw new Error(`Symlink not allowed: ${name}`);
      } catch (error) {
        if (error.code !== 'ENOENT') throw error;
        entries.set(name, null);
        return;
      }
    }
    const before = await lstat(path);
    if (before.isDirectory()) {
      const children = (await readdir(path)).sort(order);
      for (const child of children) await visit(`${name}/${child}`);
      return;
    }
    if (!before.isFile()) throw new Error(`Expected regular file: ${name}`);
    const bytes = await readFile(path);
    const after = await lstat(path);
    if (before.size !== after.size || before.mtimeMs !== after.mtimeMs || before.ino !== after.ino) {
      throw new Error(`File changed during fingerprinting: ${name}`);
    }
    entries.set(name, hash(bytes));
  }
  for (const scope of scopes) await visit(scope);
  const manifest = { schema: 1, scopes, files: [...entries].sort(([a], [b]) => order(a, b)).map(([path, sha256]) => ({ path, sha256 })) };
  return { ...manifest, fingerprint: hash(canonical(manifest)) };
}

export async function verify(root, expected) {
  if (expected?.schema !== 1 || !Array.isArray(expected.scopes) || !Array.isArray(expected.files) || hash(canonical(expected)) !== expected.fingerprint) {
    throw new Error('Invalid or corrupted manifest');
  }
  const actual = await snapshot(root, expected.scopes);
  const oldFiles = new Map(expected.files.map(item => [item.path, item.sha256]));
  const newFiles = new Map(actual.files.map(item => [item.path, item.sha256]));
  const changes = [...new Set([...oldFiles.keys(), ...newFiles.keys()])].sort(order)
    .filter(path => oldFiles.has(path) !== newFiles.has(path) || oldFiles.get(path) !== newFiles.get(path))
    .map(path => ({ path, before: oldFiles.get(path) ?? null, after: newFiles.get(path) ?? null }));
  return { valid: actual.fingerprint === expected.fingerprint, expected: expected.fingerprint, actual: actual.fingerprint, changes };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const [command, root, ...args] = process.argv.slice(2);
    if (!root || !['snapshot', 'verify'].includes(command) || (command === 'verify' && args.length !== 1)) {
      throw new Error('Usage: fingerprint.mjs snapshot ROOT SCOPE... | verify ROOT MANIFEST.json');
    }
    const result = command === 'snapshot' ? await snapshot(root, args) : await verify(root, JSON.parse(await readFile(args[0], 'utf8')));
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
    if (command === 'verify' && !result.valid) process.exitCode = 1;
  } catch (error) {
    process.stderr.write(`${error.message}\n`);
    process.exitCode = 2;
  }
}
