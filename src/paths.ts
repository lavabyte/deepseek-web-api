// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 cv-superding (Ding Li)
// Modifications Copyright 2026 deepseek-web-api contributors (see NOTICE).
/**
 * Path resolution — kept in its own file to avoid a circular import between
 * auth.ts and accounts.ts.
 *
 * Convention: all of the plugin's local state lives under
 * `${DSH_HOME || ~/.dsh}/web-login/` and **does not go into the
 * settings/credentials seam** — that is a generic configuration surface and is
 * not a good place for web-side credentials.
 */
import { homedir } from 'node:os'
import { join } from 'node:path'

/** DSH home directory (resolution order consistent with the ecosystem). */
export function resolveDshHome(): string {
  return process.env.DSH_HOME || join(homedir(), '.dsh')
}

/** The plugin's own state directory. */
export function webLoginDir(): string {
  return join(resolveDshHome(), 'web-login')
}

/**
 * Legacy single-account credential file (before 0.1.25 there was only this one
 * account). It is an account store now, but **this path must still be
 * recognised** — it is used for a one-time migration.
 */
export function legacyAuthFilePath(): string {
  return join(webLoginDir(), 'deepseek-auth.json')
}