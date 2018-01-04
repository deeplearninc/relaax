from __future__ import print_function
from __future__ import absolute_import

from . import decorated_saver


class LimitedSaver(decorated_saver.DecoratedSaver):
    def __init__(self, saver, limit):
        super(LimitedSaver, self).__init__(saver)
        self._limit = limit

    def save_checkpoint(self, checkpoint_id):
        checkpoint_ids = super(LimitedSaver, self).checkpoint_ids()
        if len(checkpoint_ids) < self._limit:
            super(LimitedSaver, self).save_checkpoint(checkpoint_id)
        else:
            least_checkpoint_id_to_keep = sorted(checkpoint_ids)[-self._limit]
            if checkpoint_id > least_checkpoint_id_to_keep:
                super(LimitedSaver, self).save_checkpoint(checkpoint_id)
                checkpoint_ids.add(checkpoint_id)
                for checkpoint_id in sorted(checkpoint_ids)[:-self._limit]:
                    super(LimitedSaver, self).remove_checkpoint(checkpoint_id)
