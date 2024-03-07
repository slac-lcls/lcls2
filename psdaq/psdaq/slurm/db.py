######################################################################################
# Simple Database for storing slurm job submitted for daq control
# Instance:
#    PK 
#   {1: slurm_job_id1, sbparms, cmd (daq CLIs), DbHistoryStatus.SUBMITTED,
#    2: slurm_job_id2, sbparms, cmd (daq CLIs(, DbHistoryStatus.REPLACED,
#
######################################################################################

class DbHistoryStatus():
    SUBMITTED = 0
    CANCELLED = 1
    REPLACED = 2
    @staticmethod
    def get_name(ID):
        info = {0: 'SUBMITTED', 1: 'CANCELLED', 2: 'REPLACED'}
        return info[ID]


class DbHistoryColumns():
    SLURM_JOB_ID = 0
    SBPARMS = 1
    CMD = 2
    STATUS = 3


class DbHelper():
    def __init__(self):
        self.instance = {} 

    def set(self, instance_id, what, val):
        self.instance[instance_id][what] = val

    def save(self, obj):
        next_id = 1
        if self.instance:
            ids = list(self.instance.keys())
            next_id = max(ids) + 1
        self.instance[next_id] = [obj['slurm_job_id'],
                obj['sbparms'],
                obj['cmd'],
                DbHistoryStatus.SUBMITTED]
        return next_id

    def get(self, instance_id):
        found_instance = None
        if instance_id in self.instance:
            found_instance = self.instance[instance_id]
        return found_instance

    def delete(self, instance_id):
        removed_value = self.instance.pop(instance_id, 'No Key found')
        print(f'delete called {removed_value=}')




