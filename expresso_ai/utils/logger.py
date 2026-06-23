import pandas as pd

class Logger():

    def __init__(self, timestamp, logfile_path):
        """
        timestamp: unique identifier of the current experiment to log. 
        logfile_path: points to the path of the logfile
        """
        self.timestamp = timestamp
        self.logfile_path = logfile_path
        self.log_df = log_df = pd.read_csv(logfile_path, index_col=[0])
        self.row = {'timestamp': timestamp}
        #Keep track of the row of experiment to modify it later
        len_df = len(self.log_df)
        self.row_index = len(self.log_df)
        self.log_df = self.log_df.append(self.row, ignore_index = True)
        assert len_df == len(self.log_df) - 1, "Expected {}, but got {}".format(len(self.log_df) -1, len_df)
        #Initialize the current experiment on the csv file
        log_df = log_df.append(self.row, ignore_index = True)
        log_df.to_csv(logfile_path)

    def log(self,  **kwargs):
        self.row.update(kwargs)
        #update the row on the dataframe and write to the csv file
        #self.log_df.iloc[self.row_index] = self.row
        for key, value in kwargs.items():
            if isinstance(value, list):
                #wrap object around a list
                #if length of object is 1, pandas won't complain
                value = [[value]]
            self.log_df.loc[self.row_index, key] = value
        #self.log_df.iloc[self.row_index] = 0
        self.log_df.to_csv(self.logfile_path)
        return
