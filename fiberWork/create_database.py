import sqlite3

class Create_Database():
    def __init__(self,):
        super().__init__()
        self.conn = sqlite3.connect('./fiber_cabling.db') 
        self.cursor = self.conn.cursor()
        self.create_database()
        
        
    def BOSLabels(self):
    # This function returns a list of labels for the BOS (Back Office System) ports.
        labels=['1.1.1', '1.1.2', '1.1.3', '1.1.4', '1.1.5', '1.1.6', '1.1.7', '1.1.8', 
                '1.2.1', '1.2.2', '1.2.3', '1.2.4', '1.2.5', '1.2.6', '1.2.7', '1.2.8',
                '1.3.1', '1.3.2', '1.3.3', '1.3.4', '1.3.5', '1.3.6', '1.3.7', '1.3.8',
                '1.4.1', '1.4.2', '1.4.3', '1.4.4', '1.4.5', '1.4.6', '1.4.7', '1.4.8',
                '1.5.1', '1.5.2', '1.5.3', '1.5.4', '1.5.5', '1.5.6', '1.5.7', '1.5.8',
                '1.6.1', '1.6.2', '1.6.3', '1.6.4', '1.6.5', '1.6.6', '1.6.7', '1.6.8',
                '1.7.1', '1.7.2', '1.7.3', '1.7.4', '1.7.5', '1.7.6', '1.7.7', '1.7.8',
                '1.8.1', '1.8.2', '1.8.3', '1.8.4', '1.8.5', '1.8.6', '1.8.7', '1.8.8',
                '2.1.1', '2.1.2', '2.1.3', '2.1.4', '2.1.5', '2.1.6', '2.1.7', '2.1.8',
                '2.2.1', '2.2.2', '2.2.3', '2.2.4', '2.2.5', '2.2.6', '2.2.7', '2.2.8',
                '2.3.1', '2.3.2', '2.3.3', '2.3.4', '2.3.5', '2.3.6', '2.3.7', '2.3.8',
                '2.4.1', '2.4.2', '2.4.3', '2.4.4', '2.4.5', '2.4.6', '2.4.7', '2.4.8',  
                '2.5.1', '2.5.2', '2.5.3', '2.5.4', '2.5.5', '2.5.6', '2.5.7', '2.5.8',
                '2.6.1', '2.6.2', '2.6.3', '2.6.4', '2.6.5', '2.6.6', '2.6.7', '2.6.8',
                '2.7.1', '2.7.2', '2.7.3', '2.7.4', '2.7.5', '2.7.6', '2.7.7', '2.7.8',
                '2.8.1', '2.8.2', '2.8.3', '2.8.4', '2.8.5', '2.8.6', '2.8.7', '2.8.8',
                '3.1.1', '3.1.2', '3.1.3', '3.1.4', '3.1.5', '3.1.6', '3.1.7', '3.1.8',
                '3.2.1', '3.2.2', '3.2.3', '3.2.4', '3.2.5', '3.2.6', '3.2.7', '3.2.8',
                '3.3.1', '3.3.2', '3.3.3', '3.3.4', '3.3.5', '3.3.6', '3.3.7', '3.3.8',
                '3.4.1', '3.4.2', '3.4.3', '3.4.4', '3.4.5', '3.4.6', '3.4.7', '3.4.8',
                '3.5.1', '3.5.2', '3.5.3', '3.5.4', '3.5.5', '3.5.6', '3.5.7', '3.5.8',
                '3.6.1', '3.6.2', '3.6.3', '3.6.4', '3.6.5', '3.6.6', '3.6.7', '3.6.8',
                '3.7.1', '3.7.2', '3.7.3', '3.7.4', '3.7.5', '3.7.6', '3.7.7', '3.7.8',
                '3.8.1', '3.8.2', '3.8.3', '3.8.4', '3.8.5', '3.8.6', '3.8.7', '3.8.8',
                '4.1.1', '4.1.2', '4.1.3', '4.1.4', '4.1.5', '4.1.6', '4.1.7', '4.1.8',
                '4.2.1', '4.2.2', '4.2.3', '4.2.4', '4.2.5', '4.2.6', '4.2.7', '4.2.8',
                '4.3.1', '4.3.2', '4.3.3', '4.3.4', '4.3.5', '4.3.6', '4.3.7', '4.3.8',
                '4.4.1', '4.4.2', '4.4.3', '4.4.4', '4.4.5', '4.4.6', '4.4.7', '4.4.8',
                '4.5.1', '4.5.2', '4.5.3', '4.5.4', '4.5.5', '4.5.6', '4.5.7', '4.5.8',
                '4.6.1', '4.6.2', '4.6.3', '4.6.4', '4.6.5', '4.6.6', '4.6.7', '4.6.8',
                '4.7.1', '4.7.2', '4.7.3', '4.7.4', '4.7.5', '4.7.6', '4.7.7', '4.7.8',
                '4.8.1', '4.8.2', '4.8.3', '4.8.4', '4.8.5', '4.8.6', '4.8.7', '4.8.8',
                '5.1.1', '5.1.2', '5.1.3', '5.1.4', '5.1.5', '5.1.6', '5.1.7', '5.1.8',
                '5.2.1', '5.2.2', '5.2.3', '5.2.4', '5.2.5', '5.2.6', '5.2.7', '5.2.8',
                '5.3.1', '5.3.2', '5.3.3', '5.3.4', '5.3.5', '5.3.6', '5.3.7', '5.3.8',
                '5.4.1', '5.4.2', '5.4.3', '5.4.4', '5.4.5', '5.4.6', '5.4.7', '5.4.8',
                '5.5.1', '5.5.2', '5.5.3', '5.5.4', '5.5.5', '5.5.6', '5.5.7', '5.5.8',
                '5.6.1', '5.6.2', '5.6.3', '5.6.4', '5.6.5', '5.6.6', '5.6.7', '5.6.8',
                '5.7.1', '5.7.2', '5.7.3', '5.7.4', '5.7.5', '5.7.6', '5.7.7', '5.7.8',
                '5.8.1', '5.8.2', '5.8.3', '5.8.4', '5.8.5', '5.8.6', '5.8.7', '5.8.8',
                '6.1.1', '6.1.2', '6.1.3', '6.1.4',
                ]
        return labels
    
    def DetectorLabels(self):
    # This function returns a list of labels for the Detector ports.
        labels={}
        labels['TMO']=["atm_opal_ip1_0",
            "atm_piranha_ip1_0",
            "atm_piranha_ip2_0",
            "dream_hsd_lmcp_0",
            "epix100_0",
            "fzp_opal_0",
            "fzp_piranha_0",
            "hsd_peppex_0",
            "hsd_peppex_1",
            "hsd_test_0",
            "tof_hsd_0",
            "laser_wav8_0",
            "peppex_opal_0",
            "timing_0",
            "timing_1",
            "tmo_fim0_0",
            "tmo_fim1_0",
            "tmo_opal1_0",
            "tmo_opal2_0",
            "tmo_opal3_0",
            "tmo_piranha_0",
            "tmoopal2_0",
            "tmoopal_0",
            "trigger_0",
            "txi_fim1_0",
            ]
        for i in range(0,20): labels['TMO'].append(f'hsd_{i}')
        labels['RIX']=[
            'c_atmopal_0',  
            'c_epixm_0',
            'c_piranha_0',
            'crix_w8_0',
            'crixs_las_w8_0',
            'epixhr_0',
            'epixm_0',
            'epixuhr_0',
            'fzp_piranha_0',
            'mono_hrencoder_0',
            'piranha_0_0',
            'q_atmopal_0',
            'q_piranha_0',
            'qrix_w8_0',
            'rix_fim0_0',
            'rix_fim1_0',
            'rix_fim2_0',
            'rix_fim3_0',
            'rpixhr_0',
            'timing_0',
            'timing_1',
            'trigger_0',
        ]
        for i in range(0, 6): labels['RIX'].append(f'hsd_{i}')
        return labels
    
    def addContainer(self, parent, name):
            # This function adds a new container to the database with a specified parent and name.
            self.cursor.execute(f"INSERT INTO Container (idParent, ContainerName) VALUES ('{parent}', '{name}')")
            self.conn.commit()  
            return self.cursor.lastrowid          
            
    def addPort(self, idParent, N):
        # This function adds ports to a specified parent container in the database.
        for n in range(1, N+1, 2):
            self.cursor.execute(f"INSERT INTO Port (idParent, PortName) VALUES ('{idParent}', '{n:02d}-{n+1:02d}')")
            self.conn.commit()
            
    def addPortLabels(self, idParent, labels):
        # This function adds multiple ports with specified labels to a parent container in the database.
        for label in labels:
            self.cursor.execute(f"INSERT INTO Port (idParent, PortName) VALUES ('{idParent}', '{label}')")
        self.conn.commit()
        return self.cursor.lastrowid

    def create_database(self):
            
        # This function creates a demo data structure in the SQLite database for setup purposes.
        
        self.cursor.execute("CREATE TABLE IF NOT EXISTS Container ( idContainer INTEGER PRIMARY KEY AUTOINCREMENT, ContainerName VARCHAR(45),idParent INTEGER);")
        self.cursor.execute("CREATE TABLE IF NOT EXISTS Port ( idPort INTEGER PRIMARY KEY AUTOINCREMENT, PortName VARCHAR(45), idParent INTEGER, idin INTEGER, idout INTEGER);")
        self.cursor.execute("CREATE TABLE IF NOT EXISTS Connections (idin INTEGER, idout INTEGER, Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP);")
        self.conn.commit()
        
        Containers = ['TMO', 'RIX', '208']
        DetectorsLabels = self.DetectorLabels()
        for container in Containers:    
            idc = self.addContainer(0, container)
            if container in DetectorsLabels:
                i = self.addContainer(idc, 'Detectors')
                idport = self.addPortLabels(i, DetectorsLabels[container])
                
            if container == '208':
                id208 = idc
            for rack in range(1, 4):
                idr = self.addContainer(idc, f'Rack{rack}')
                for patch in range(1, 5):
                    idp = self.addContainer(idr, f'FODU{patch}')
                    for cassette in range(1, 5):
                        idcas = self.addContainer(idp, f'Cassette{cassette}')
                        self.addPort(idcas, 36)
        idBOS = self.addContainer(id208, 'BOS')
        self.addPortLabels(idBOS, self.BOSLabels())

        MPOs = ['Top_Patch_Panel', 'Bottom_Patch_Panel_MPO']
        for mpo in MPOs:
            idmpo = self.addContainer(id208, mpo)
            for col in range(1,13):
                for row in range(1,4):
                    idbreakc = self.addContainer(idmpo, f'{col:02d}-{row:02d}' )
                    self.addPortLabels(idbreakc, ['Lane1', 'Lane2', 'Lane3', 'Lane4'])
        iddrp=self.addContainer(0, 'SRCF')
        for i in range(1,60):
            idcmp=self.addContainer(iddrp, f'cmp_{i:03d}')
            id=self.addContainer(idcmp, 'Datadev_0')
            id=self.addContainer(idcmp, 'Datadev_1')
        
        MPOs = ['Top_Patch_Panel', 'Bottom_Patch_Panel_MPO']
        for mpo in MPOs:
            idmpo = self.addContainer(iddrp, mpo)
            for col in range(1,13):
                for row in range(1,4):
                    idbreakc = self.addContainer(idmpo, f'{col:02d}-{row:02d}' )
                    self.addPortLabels(idbreakc, ['Lane1', 'Lane2', 'Lane3', 'Lane4'])


                
if __name__ == '__main__':
    Create = Create_Database()