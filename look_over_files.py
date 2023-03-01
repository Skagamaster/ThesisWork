"""
This macro is used to see if we have zombie files in order to
murder them so that we can properly hadd stuff.
"""

import os
import uproot as up

os.chdir(r'F:\AuAu200\Dagger2')

files = os.listdir()
badfiles = []
for file in files:
    try:
        arr = up.open(file)['PicoDst']
    except Exception as e:
        badfiles.append(file)

print("Total bad:", (len(badfiles)/len(files))*100, r'%')
"""
print("And now, young badfiles ... you will die.")
for i in badfiles:
    os.remove(i)  # Danger zone coding right here.
"""
