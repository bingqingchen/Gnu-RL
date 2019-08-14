from gym.envs.registration import register
import os
import fileinput

FD = os.path.dirname(os.path.realpath(__file__));

register(
         id='5Zone-sim_TMY2-v0',
         entry_point='eplus_env.envs:EplusEnv',
         kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/', # The EnergyPlus software path
         'weather_path':FD + '/envs/weather/pittsburgh_TMY2.epw', # The epw weather file
         'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
         'variable_path':FD + '/envs/eplus_models/5Zone/variables_Default.cfg', # The cfg file
         'idf_path':FD + '/envs/eplus_models/5Zone/5Zone_Default.idf', # The idf file
         'env_name': '5Zone-sim-v0',
         });

register(
         id='5Zone-sim_TMY3-v0',
         entry_point='eplus_env.envs:EplusEnv',
         kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/', # The EnergyPlus software path
         'weather_path':FD + '/envs/weather/pittsburgh_TMY3.epw', # The epw weather file
         'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
         'variable_path':FD + '/envs/eplus_models/5Zone/variables_Default.cfg', # The cfg file
         'idf_path':FD + '/envs/eplus_models/5Zone/5Zone_Default.idf', # The idf file
         'env_name': '5Zone-sim-v0',
         });

register(
         id='5Zone-control_TMY2-v0',
         entry_point='eplus_env.envs:EplusEnv',
         kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/', # The EnergyPlus software path
         'weather_path':FD + '/envs/weather/pittsburgh_TMY2.epw', # The epw weather file
         'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
         'variable_path':FD + '/envs/eplus_models/5Zone/variables_Control.cfg', # The cfg file
         'idf_path':FD + '/envs/eplus_models/5Zone/5Zone_Control.idf', # The idf file
         'env_name': '5Zone-control-v0',
         });

register(
         id='5Zone-control_TMY3-v0',
         entry_point='eplus_env.envs:EplusEnv',
         kwargs={'eplus_path':FD + '/envs/EnergyPlus-8-6-0/', # The EnergyPlus software path
         'weather_path':FD + '/envs/weather/pittsburgh_TMY3.epw', # The epw weather file
         'bcvtb_path':FD + '/envs/bcvtb/', # The BCVTB path
         'variable_path':FD + '/envs/eplus_models/5Zone/variables_Control.cfg', # The cfg file
         'idf_path':FD + '/envs/eplus_models/5Zone/5Zone_Control.idf', # The idf file
         'env_name': '5Zone-control-v0',
         });





