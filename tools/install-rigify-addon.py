"""Run with normal background Blender (no factory-startup) to preserve preferences."""
from datetime import datetime
from pathlib import Path
import shutil
import zipfile

import addon_utils
import bpy

project = Path(__file__).resolve().parents[1]
stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
config = Path(bpy.utils.user_resource('CONFIG'))
preferences = config / 'userpref.blend'
if preferences.exists():
    backup = config / f'userpref.before-framecraft-rigify-{stamp}.blend'
    shutil.copy2(preferences, backup)
    print('PREFERENCES_BACKUP', backup)
addons = Path(bpy.utils.user_resource('SCRIPTS', path='addons', create=True))
destination = addons / 'framecraft_rigify.py'
if destination.exists():
    shutil.copy2(destination, destination.with_suffix(f'.before-{stamp}.py.bak'))
shutil.copy2(project / 'framecraft_rigify.py', destination)
bpy.utils.refresh_script_paths()
for name in ('rigify', 'framecraft_rigify'):
    if not addon_utils.check(name)[1]:
        addon_utils.enable(name, default_set=True, persistent=True)
    assert addon_utils.check(name)[1], f'Could not enable {name}'
assert hasattr(bpy.types.Scene, 'framecraft_rigify')
bpy.ops.wm.save_userpref()
documents = Path.home() / 'Documents'
archive = documents / 'Framecraft Rigify Retarget.zip'
if archive.exists():
    shutil.copy2(archive, archive.with_name(f'{archive.stem}.before-{stamp}.zip'))
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
    output.write(project / 'framecraft_rigify.py', 'framecraft_rigify.py')
    output.write(project / 'RIGIFY.md', 'RIGIFY.md')
print('RIGIFY_ADDON_INSTALLED', destination)
print('RIGIFY_ADDON_ARCHIVE', archive)
