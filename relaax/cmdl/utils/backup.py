from __future__ import print_function

from builtins import object

import os
import glob
import shutil
import string


class Backup(object):

    def __init__(self, filename):
        self.filename = filename

    def make_backup(self):
        """Save a numbered backup file or folder.

        If the file or folder doesn't already exist, there's nothing to do.
        """
        new_name = self._versioned_name(self._current_revision() + 1)
        if os.path.isfile(self.filename):
            shutil.copy2(self.filename, new_name)
            return (True, new_name)
        if os.path.isdir(self.filename):
            os.rename(self.filename, new_name)
            return (True, new_name)
        return (False, None)

    def _versioned_name(self, revision):
        """Get filename with a revision number appended."""
        return "%s.~%s~" % (self.filename, revision)

    def _current_revision(self):
        """Get the revision number of the largest existing backup."""
        revisions = [0] + self._revisions()
        return max(revisions)

    def _revisions(self):
        """Get the revision numbers of all backups."""
        revisions = []
        backup_names = glob.glob("%s.~[0-9]*~" % (self.filename))
        for name in backup_names:
            try:
                revision = int(string.split(name, "~")[-2])
                revisions.append(revision)
            except ValueError:
                # Some ~[0-9]*~ extensions may not be wholly numeric.
                pass
        revisions.sort()
        return revisions
