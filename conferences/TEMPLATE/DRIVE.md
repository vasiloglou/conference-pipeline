# Conference Drive Backup

Replace this file when creating a real conference workspace.

- Drive folder: `TODO`
- Drive leaf: `TODO`
- Mounted path, when available: `TODO`
- Sync profile: `backup`
- Manifest: `storage/storage_manifest.json`

Use Google Drive as the backup location for bulky generated artifacts. Do not
run a full local restore unless the user explicitly asks for one. When an
artifact is missing locally, read the manifest first and retrieve only the
needed file, directory, or artifact family from the documented Drive leaf.
