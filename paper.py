# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 15:12:37 2025

@author: Cyril Liu
"""

import numpy as np
import scipy.stats as st
import matplotlib.cm as cm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import ListedColormap

file_path_data_sheet = 'data_sheet.xlsx'
df = pd.DataFrame(pd.read_excel(file_path_data_sheet))
cp_params = {"resolution_s": 50, 
             "resolution_c": 50,
             "min_storage": 150,
             "max_storage": 3000, 
             "min_consum": 50, 
             "max_consum":1500,
             "sampling_times":10000}

def remake_df(df):
    date_series = df["date_month"]
    inflow_series = df["inflow"]
    storage_series = df["storage"]
    consumption_series = df["總引水量(C)"]
    rfd_series = df["RFD*"]
    threshold_series = df["RFD_his"]
    cwdi_series = df["WDI*"]
    pre_storage_series = df["initial_storage"]    
    drought_record_series = df["乾旱紀錄"]
    del_storage_series = df["del_storage"]
    change_series = df["change"]
    dis_charge_series = df["discharge_power"]
    ev_series = df["E"]
    a_plus_storage_series = df["inflow+storage"]
    bier_l_series = df["BIER_L"]
    storage_correct_series = del_storage_series - change_series
    df_out = pd.DataFrame()
    df_out.insert(loc = 0, column = "date", value = date_series)
    df_out.insert(loc = 1, column = "inflow", value = inflow_series)
    df_out.insert(loc = 2, column = "pre_storage", value = pre_storage_series)
    df_out.insert(loc = 3, column = "storage", value = storage_series)
    df_out.insert(loc = 4, column = "consumption", value = consumption_series)
    df_out.insert(loc = 5, column = "rfd", value = rfd_series)
    df_out.insert(loc = 6, column = "threshold",value = threshold_series)
    df_out.insert(loc = 7, column = "cwdi", value = cwdi_series)
    df_out.insert(loc = 8, column = "drought_record", value = drought_record_series)
    df_out.insert(loc = 9, column = "del_storage", value = del_storage_series)
    df_out.insert(loc = 10, column = "change", value = change_series)
    df_out.insert(loc = 11, column = "storage_correct", value = storage_correct_series)
    df_out.insert(loc = 12, column = "discharge_power", value = dis_charge_series)
    df_out.insert(loc = 13, column = "ev", value = ev_series)
    df_out.insert(loc = 14, column = "inflow+storage", value = a_plus_storage_series)
    df_out.insert(loc = 15, column = "bier_l", value = bier_l_series)
    df_out = df_out.dropna()
    df_out = df_out.reset_index(drop = True)
    return df_out       

class hydraulic_freq_analyse:
    
    def __init__(self,data_list , log_list = True):
        self.data_list = data_list
        array_list = []
        property_list = []
        
        if log_list == True:
            distribution_list =  ['norm', 'gumbel_r', 'gamma',  'pearson3', 'lognorm', "gumbel_l"]
        else :
            distribution_list =  ['norm', 'gumbel_r', 'gamma',  'pearson3', "gumbel_l"]
      
        for i in range(len(data_list)):
            temp = np.asarray(data_list[i])
            array_list.append(temp)
            temp_mean = temp.mean()
            temp_var = temp.var()
            temp_skew = st.skew(temp)
            temp_kurt = st.kurtosis(temp)
            temp_property = np.asarray([temp_mean, temp_var, temp_skew, temp_kurt])
            property_list.append(temp_property)
        
        self.array_list = array_list
        self.property_list = property_list
        self.distribution_list = distribution_list

    def get_norm_param(self, sample):
        output = st.norm.fit(sample)
        return output 
    
    def get_gumbel_r_param(self, sample):
        output = st.gumbel_r.fit(sample)
        return output
    
    def get_gumbel_l_param(self, sample):
        output = st.gumbel_l.fit(sample)
        return output
    
    def get_pearson3_param(self, sample):
        output = st.pearson3.fit(sample)
        return output
    
    def get_gamma_param(self, sample):
        output = st.gamma.fit(sample)
        return output
    
    def get_lognorm_param(self, sample):
        output = st.lognorm.fit(sample)
        return output
    
    def get_loggamma_param(self, sample):
        output = st.loggamma.fit(sample)
        return output
    
    def ktest(self, sample, pick_distribution):
        if pick_distribution == 'norm':
            
            parameter = self.get_norm_param(sample)
            return st.kstest(sample, 'norm', parameter)[1], parameter
        
        elif pick_distribution == 'gamma':
            
            parameter = self.get_gamma_param(sample)
            return st.kstest(sample, 'gamma',parameter)[1], parameter
        
        elif pick_distribution == 'gumbel_r':
            
            parameter = self.get_gumbel_r_param(sample)
            return st.kstest(sample, 'gumbel_r', parameter)[1], parameter
        
        elif pick_distribution == 'gumbel_l':
            
            parameter = self.get_gumbel_l_param(sample)
            return st.kstest(sample, 'gumbel_l', parameter)[1], parameter
        
        elif pick_distribution == 'lognorm':
            
            parameter = self.get_lognorm_param(sample)
            return st.kstest(sample, 'lognorm', parameter)[1], parameter
        
        elif pick_distribution == 'pearson3':

            parameter = self.get_pearson3_param(sample)
            return st.kstest(sample, 'pearson3', parameter)[1], parameter

        elif pick_distribution == 'loggamma':
            
            parameter = self.get_loggamma_param(sample)
            return st.kstest(sample, 'loggamma', parameter)[1], parameter   
        
    def get_suitable_distribution_list(self):
        array_list = self.array_list
        distribution_list = self.distribution_list
        result = []
        for i in range(len(array_list)):
            compare_list = []
            sample = array_list[i]
            temp_parameter_list = []
            for j in range(len(distribution_list)):
                
                temp = self.ktest(sample, distribution_list[j])[0]
                temp_p = self.ktest(sample, distribution_list[j])[1]
                compare_list.append(temp)
                temp_parameter_list.append(temp_p)
            
            compare_list = np.asarray(compare_list)
            maxx = np.max(compare_list)
            k = np.argmax(compare_list)

            temp_result = distribution_list[k]
            temp_parameter = temp_parameter_list[k]
            result.append([temp_result,temp_parameter,maxx])

        return result

class standardized_process:

    def __init__(self, df_r, name):
        self.data_series = df_r[name]
        self.month_list = self.get_month_data(df_r, name)
        self.distri_info = hydraulic_freq_analyse(self.month_list).get_suitable_distribution_list()
        self.date_series = df_r["date"]
    
    def get_month_data(self, df_r, name):
        month_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
    
        for i in range(len(df_r)):
            for j in range(1,13):
                if df_r['date'][i].month == j:
                    month_list[j-1].append(df_r[name][i])
            
        for i in range(len(month_list)):
            temp_month = np.asarray(month_list[i],dtype = np.float64)
            temp_month.sort()
            month_list[i] = temp_month
            
        return month_list
    
    def get_ecdf(self, data, month):
        
        rank_pool = self.month_list[month - 1]
        for i in range(len(rank_pool)):
            if data == rank_pool[i]:
                rank = i + 1
                break
            else:
                continue
        return st.norm.ppf((rank - 0.44) / (len(rank_pool) + 0.12))
    
    def transformation_v1(self):
        output = []
        for i in range(len(self.data_series)):
            temp_output = self.get_ecdf(self.data_series[i],
                                        self.date_series[i].month)
            output.append(temp_output)
        return np.array(output)

    def transformation_v2(self):
        output = []
        mean_month = []
        std_month = []
        for i in range(12):
            mean_month.append(np.mean(self.month_list[i]))
            std_month.append(np.std(self.month_list[i]))
        
        for i in range(len(self.date_series)):
            month = self.date_series[i].month
            temp_output = (self.data_series[i] - mean_month[month - 1]) / std_month[month - 1]
            output.append(temp_output)
            
        return np.array(output)

class pair_modeling:
    
    def __init__(self, array_list):
        
        self.array_list = array_list
        self.array_mean_list = np.mean(array_list, axis = 2)
        regress_params_list = []
        noise_list = []
        pearsonr_list =[]
        for i in range(12):
            regress_params_list.append(self.get_month_regression_storage_to_correction(i+1))
            noise_list.append(self.regression_noise(i+1))
            pearsonr_list.append(self.get_pearsonr(i+1))
            
        self.regress_params = regress_params_list
        self.noise_list = noise_list
        self.pearsonr_list = pearsonr_list
        
        hyf = hydraulic_freq_analyse(noise_list)
        self.noise_distribution = hyf.get_suitable_distribution_list()
        
    def get_pearsonr(self, month):
        
        x = self.array_list[1][month - 1]
        y = self.array_list[0][month - 1]
        
        return st.pearsonr(x,y)[0]
    
    def get_month_regression_storage_to_correction(self, month):

        x = self.array_list[1][month - 1]
        y = self.array_list[0][month - 1]        
        slope, intercept, r, p, se = st.linregress(x, y)
        
        return  slope, intercept
    
    def regression_noise(self, month):

        x = self.array_list[1][month - 1]
        y = self.array_list[0][month - 1]
        slope, intercept = self.get_month_regression_storage_to_correction(month)
        y_p = slope * x + intercept
        noise = -1 * abs(y-y_p)
        return noise

class conditional_probability_analysis:
    
    def __init__(self, df_r):
        
        self.inflow = self.get_month_data(df_r, "inflow")
        self.ev = self.get_month_data(df_r, "ev")
        self.a_plus_storage = self.get_month_data(df_r, "inflow+storage")
        self.df_r = df_r
        self.inflow_info = hydraulic_freq_analyse(self.inflow).get_suitable_distribution_list()
        self.ev_info = hydraulic_freq_analyse(self.ev).get_suitable_distribution_list()

        mean_discharge_array, discharge_inflow_pair = self.get_monthly_discharge_pair()
        self.discharge_array = mean_discharge_array
        self.discharge_inflow_pair = discharge_inflow_pair
        discharge_modeling = pair_modeling(self.discharge_inflow_pair)
        self.id_noise_info = discharge_modeling.noise_distribution
        self.id_regress_info = discharge_modeling.regress_params

        mean_correct_array, correct_pair = self.get_monthly_correction_pair()
        self.storage_correct_array = mean_correct_array
        self.correct_pair = correct_pair
        storage_correct = pair_modeling(self.correct_pair)
        self.sc_noise_info = storage_correct.noise_distribution
        self.sc_regress_info = storage_correct.regress_params
        
        risk_map_rfd, risk_map_cwdi, risk_map_fail, risk_map_sucess = self.get_annual_conditional_probability()
        self.risk_map_fail = risk_map_fail
        self.risk_map_sucess = risk_map_sucess
        self.risk_map_rfd = risk_map_rfd
        self.risk_map_cwdi = risk_map_cwdi
        self.storage_index_standard = np.array(np.linspace(cp_params["min_storage"],
                                                      cp_params["max_storage"], num = cp_params["resolution_s"]))
        self.consum_index_standard = np.array(np.linspace(cp_params["min_consum"],
                                                      cp_params["max_consum"], num = cp_params["resolution_c"]))

    def get_threshold(self):
        return np.array(self.df_r["threshold"][:12])
    
    def get_bier(self):
        return np.array(self.df_r["bier_l"][:12])

    def storage_change_correction_sampling(self, month, inflow):
        slope, intercept = self.sc_regress_info[month - 1]
        noise = self.sampling(month, len(inflow), self.sc_noise_info)     
        return (inflow * slope + intercept  + noise)

    def discharge_sampling(self, month, inflow):
        slope, intercept = self.id_regress_info[month - 1]
        noise = self.sampling(month, len(inflow), self.sc_noise_info)     
        return (inflow * slope + intercept  + noise)
        
    def sampling(self, month ,size, distribution_info):
        info = distribution_info[month - 1]
        name = info[0]
        dis_param = info[1]
        if name == 'norm':
            return st.norm.rvs(size = size,
                               loc = dis_param[0],
                               scale = dis_param[1])
        elif name == 'gamma':
            return st.gamma.rvs(size = size, 
                                a = dis_param[0],
                                loc = dis_param[1], 
                                scale = dis_param[2])
        elif name == 'pearson3':
            return st.pearson3.rvs(size = size,  
                                   skew = dis_param[0],
                                   loc = dis_param[1],
                                   scale = dis_param[2])
        elif name == 'gumbel_r':
            return st.gumbel_r.rvs(size = size,
                                   loc = dis_param[0],
                                   scale = dis_param[1])
        elif name == 'gumbel_l':
            return st.gumbel_l.rvs(size = size,
                                   loc = dis_param[0],
                                   scale = dis_param[1])
        elif name == 'lognorm':
            return st.lognorm.rvs(size = size,
                                  s = dis_param[0],
                                  loc = dis_param[1],
                                  scale = dis_param[2])
        elif name == 'loggamma':
            return st.loggamma.rvs(size = size,
                                   c = dis_param[0],
                                   loc = dis_param[1], 
                                   scale = dis_param[2])
    
    def get_month_data(self, df_r, name):
        month_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
        for i in range(len(df_r)):
            for j in range(1,13):
                if df_r['date'][i].month == j:
                    month_list[j-1].append(df_r[name][i])
                else:
                    continue
        for i in range(len(month_list)):
            month_list[i] = np.asarray(month_list[i],dtype = np.float64)
        
        return month_list

    def get_monthly_correction_pair(self):
        df_r = self.df_r    
        df_rr = pd.DataFrame()
        capacity = 3000
        overflow = df_r["pre_storage"] + df_r["inflow"] - df_r["consumption"] - df_r["discharge_power"] - capacity
        overflow[overflow < 0] = 0        
        df_rr.insert(loc = 0, column = "date", value = df_r["date"])
        df_rr.insert(loc = 1, column = "inflow", value = df_r["inflow"])
        df_rr.insert(loc = 2, column = "discharge", value = df_r["discharge_power"])
        df_rr.insert(loc = 3, column = "storage", value = df_r["pre_storage"])
        df_rr.insert(loc = 4, column = "consum", value = df_r["consumption"])
        df_rr.insert(loc = 5, column = "overflow", value = overflow)
        correct = df_r["del_storage"] - (df_r["inflow"] - df_r["consumption"] - df_r["discharge_power"] - df_rr["overflow"])
        df_rr.insert(loc = 6, column = "correct", value = correct)

        df_rr = df_rr[:][0:336]
        month_list_correction = [[],[],[],[],[],[],[],[],[],[],[],[]]
        month_list_inflow = [[],[],[],[],[],[],[],[],[],[],[],[]]

        for i in range(len(df_rr)):
            for j in range(len(month_list_correction)):
                if df_rr["date"][i].month == j + 1:                  
                    month_list_correction[j].append(df_rr["correct"][i])
                    month_list_inflow[j].append(df_rr["inflow"][i])
                else:
                    continue
        
        for i in range(len(month_list_correction)):
            month_list_correction[i] = np.array(month_list_correction[i])
            month_list_inflow[i] = np.array(month_list_inflow[i])

        return np.mean(month_list_correction, axis = 1), (month_list_correction, month_list_inflow)

    def get_monthly_discharge_pair(self):
        df_r = self.df_r    
        month_list_correction = [[],[],[],[],[],[],[],[],[],[],[],[]]
        month_list_inflow = [[],[],[],[],[],[],[],[],[],[],[],[]]
        df_rr = df_r[:][0:336]
        for i in range(len(df_rr)):
            for j in range(len(month_list_correction)):
                if df_rr["date"][i].month == j + 1:                  
                    month_list_correction[j].append(df_rr["discharge_power"][i])
                    month_list_inflow[j].append(df_rr["inflow"][i])
                else:
                    continue
        
        for i in range(len(month_list_correction)):
            month_list_correction[i] = np.array(month_list_correction[i])
            month_list_inflow[i] = np.array(month_list_inflow[i])
        
        return np.mean(month_list_correction, axis = 1), (month_list_correction, month_list_inflow)


    def modified_wave_plus_sampling(self, month, size, consumption, initial_storage,
                                    bier, 
                                    capacity = cp_params["max_storage"], 
                                    min_storage = cp_params["min_storage"], alpha = 19.85):
        
        a = self.sampling(month, size, self.inflow_info)
        ev = self.sampling(month, size, self.ev_info)
        discharge = self.discharge_sampling(month, a)
        correction = self.storage_change_correction_sampling(month, a)
        storage = initial_storage + a - discharge + correction - consumption
        storage[storage > 3000] = capacity
        storage[storage < 0] = min_storage
        mean_storage = (storage + initial_storage) / 2
        
        cta = consumption / (a + mean_storage)
        consumption_max = a + initial_storage
        consumption_max[consumption_max > consumption] = consumption
        consumption = consumption_max
        cwdi = np.round(np.reciprocal((1 + 99 * np.exp(-1* alpha * cta))) , 5)
        rfd = (consumption - ev * bier) * cwdi
        
        return cwdi,rfd
    
    def check_greater_ratio(self, threshold_1, array_1, threshold_2, array_2):

        array_1_bool = array_1 > threshold_1
        array_1_int = array_1_bool.astype(int)
        array_2_bool = array_2 > threshold_2
        array_2_int = array_2_bool.astype(int)
        array_3 = array_1_int + array_2_int
        array_3_bool = array_3 >= 2
        array_3_int = array_3_bool.astype(int)
        
        return np.mean(array_1_int), np.mean(array_2_int), np.mean(array_3_int)

    def check_smaller_ratio(self, threshold_1, array_1, threshold_2, array_2):
        array_1_bool = array_1 <= threshold_1
        array_1_int = array_1_bool.astype(int)
        array_2_bool = array_2 <= threshold_2
        array_2_int = array_2_bool.astype(int)
        array_3 = array_1_int + array_2_int
        array_3_bool = array_3 >= 2
        array_3_int = array_3_bool.astype(int)
        
        return np.mean(array_1_int), np.mean(array_2_int), np.mean(array_3_int)
    
    def monthly_conditional_probability(self, month, threshold, recovery_criteria, bier ,
                                        size = cp_params["sampling_times"], 
                                        min_storage = cp_params["min_storage"], 
                                        max_storage = cp_params["max_storage"], 
                                        min_consum = cp_params["min_consum"], 
                                        max_consum = cp_params["max_consum"], 
                                        resolution_c = cp_params["resolution_c"],
                                        resolution_s = cp_params["resolution_s"]):
        
        consumption_array = np.linspace(min_consum, max_consum, resolution_c)
        storage_array = np.linspace(min_storage, max_storage, resolution_s)
        
        output_matrix_rfd = np.zeros(shape = (resolution_s, resolution_c))
        output_matrix_cwdi = np.zeros(shape = (resolution_s, resolution_c))
        output_matrix_fail = np.zeros(shape = (resolution_s, resolution_c))
        output_matrix_sucess = np.zeros(shape = (resolution_s, resolution_c))
        for i in range(len(storage_array)):
            for j in range(len(consumption_array)):
                consumption = consumption_array[j]
                initial_storage = storage_array[i]
                cwdi, rfd = self.modified_wave_plus_sampling(month, size,
                                                             consumption, 
                                                             initial_storage, bier)
                cwdi_fail, rfd_fail, fail = self.check_greater_ratio(recovery_criteria,
                                                                     cwdi,
                                                                     threshold,
                                                                     rfd)
                cwdi_sucess, rfd_sucess, sucess = self.check_smaller_ratio(recovery_criteria,
                                                                     cwdi,
                                                                     threshold,
                                                                     rfd)
                output_matrix_rfd[i][j] = rfd_fail
                output_matrix_cwdi[i][j] = cwdi_fail
                output_matrix_fail[i][j] = fail
                output_matrix_sucess[i][j] = sucess
        return output_matrix_rfd, output_matrix_cwdi, output_matrix_fail, output_matrix_sucess
    
    def get_annual_conditional_probability(self):
        bier_list = self.get_bier()
        threshold_list = self.get_threshold()
        output_sucess = []
        output_fail = []
        output_rfd_f = []
        output_cwdi_f = []
        for i in range(len(bier_list)):
            rfd, cwdi, fail, sucess = self.monthly_conditional_probability(i + 1,
                                                                           threshold_list[i],
                                                                           0.85,
                                                                           bier_list[i])
            output_sucess.append(sucess)
            output_fail.append(fail)
            output_cwdi_f.append(cwdi)
            output_rfd_f.append(rfd)
        
        return np.array(output_rfd_f), np.array(output_cwdi_f), np.array(output_fail), np.array(output_sucess)
    
    def transform_probability_dynamics(self):
        df_r = self.df_r
        date_series = df_r["date"]
        pre_storage_series = df_r["pre_storage"]
        consum_series = df_r["consumption"]
        ps_series = []
        pf_series = []
        for i in range(len(date_series)):
            month = date_series[i].month
            c_i = np.argmin(abs(self.consum_index_standard - consum_series[i]))
            s_i = np.argmin(abs(self.storage_index_standard - pre_storage_series[i]))
            ps_series.append(self.risk_map_sucess[month - 1][s_i][c_i])
            pf_series.append(self.risk_map_fail[month - 1][s_i][c_i])
        
        return np.array(pf_series), np.array(ps_series)
    
    def decomposition_p_dynamic_sucess_v1(self, start, end):
        # we preasent the order of the decomposition, t -> ws -> c 
        df_r = self.df_r
        date_series = df_r["date"][start: end + 1]
        pre_storage_series = df_r["pre_storage"][start: end + 1] 
        consum_series = df_r["consumption"][start: end + 1]
        ps = self.risk_map_sucess
        
        init_c_i = np.argmin(abs(self.consum_index_standard - consum_series[start]))
        fin_c_i = np.argmin(abs(self.consum_index_standard - consum_series[end]))
        init_s_i = np.argmin(abs(self.consum_index_standard - pre_storage_series[start]))
        fin_s_i = np.argmin(abs(self.storage_index_standard - pre_storage_series[end]))
        init_s_c_i = np.argmin(abs(self.storage_index_standard - (pre_storage_series[start] - consum_series[start])))
        init_t_i = date_series[start].month - 1
        fin_t_i = date_series[end].month - 1 
        
        season_change = ps[fin_t_i][init_s_i][init_c_i] - ps[init_t_i][init_s_i][init_c_i]
        storage_change = ps[fin_t_i][fin_s_i][init_c_i] - ps[fin_t_i][init_s_i][init_c_i]
        pre_storage_consum_change = ps[fin_t_i][init_s_i][init_c_i] - ps[fin_t_i][init_s_c_i][init_c_i]
        consum_change = ps[fin_t_i][fin_s_i][fin_c_i] - ps[fin_t_i][fin_s_i][init_c_i]
        
        anthro_press = consum_change + pre_storage_consum_change
        vis_major = season_change + storage_change - pre_storage_consum_change
        return anthro_press, vis_major, anthro_press + vis_major
    
    def decomposition_p_dynamic_fail_v1(self, start, end):
        # we preasent the order of the decomposition, t -> ws -> c 
        df_r = self.df_r
        date_series = df_r["date"][start: end + 1]
        pre_storage_series = df_r["pre_storage"][start: end + 1]
        consum_series = df_r["consumption"][start: end + 1]
        pf = self.risk_map_fail
        
        init_c_i = np.argmin(abs(self.consum_index_standard - consum_series[start]))
        fin_c_i = np.argmin(abs(self.consum_index_standard - consum_series[end]))
        init_s_i = np.argmin(abs(self.consum_index_standard - pre_storage_series[start]))
        fin_s_i = np.argmin(abs(self.storage_index_standard - pre_storage_series[end]))
        init_s_c_i = np.argmin(abs(self.storage_index_standard - (pre_storage_series[start] - consum_series[start])))
        init_t_i = date_series[start].month - 1
        fin_t_i = date_series[end].month - 1 
        
        season_change = pf[fin_t_i][init_s_i][init_c_i] - pf[init_t_i][init_s_i][init_c_i]
        storage_change = pf[fin_t_i][fin_s_i][init_c_i] - pf[fin_t_i][init_s_i][init_c_i]
        pre_storage_consum_change = pf[fin_t_i][init_s_i][init_c_i] - pf[fin_t_i][init_s_c_i][init_c_i]
        consum_change = pf[fin_t_i][fin_s_i][fin_c_i] - pf[fin_t_i][fin_s_i][init_c_i]
        
        anthro_press = consum_change + pre_storage_consum_change
        vis_major = season_change + storage_change - pre_storage_consum_change
        return anthro_press, vis_major, anthro_press + vis_major

    def decomposition_p_dynamic_sucess_v2(self, date_series, consum_series, pre_storage_series):
        # we preasent the order of the decomposition, t -> ws -> c 
        
        date_series = date_series.reset_index(drop = True)
        ps = self.risk_map_sucess
        init_c_i = np.argmin(abs(self.consum_index_standard - consum_series[0]))
        fin_c_i = np.argmin(abs(self.consum_index_standard - consum_series[-1]))
        init_s_i = np.argmin(abs(self.consum_index_standard - pre_storage_series[0]))
        fin_s_i = np.argmin(abs(self.storage_index_standard - pre_storage_series[-1]))
        init_s_c_i = np.argmin(abs(self.storage_index_standard - (pre_storage_series[0] - consum_series[0])))
        init_t_i = date_series[0].month - 1
        fin_t_i = date_series[len(date_series)-1].month - 1 
        
        season_change = ps[fin_t_i][init_s_i][init_c_i] - ps[init_t_i][init_s_i][init_c_i]
        storage_change = ps[fin_t_i][fin_s_i][init_c_i] - ps[fin_t_i][init_s_i][init_c_i]
        pre_storage_consum_change = ps[fin_t_i][init_s_i][init_c_i] - ps[fin_t_i][init_s_c_i][init_c_i]
        consum_change = ps[fin_t_i][fin_s_i][fin_c_i] - ps[fin_t_i][fin_s_i][init_c_i]
        
        anthro_press = consum_change + pre_storage_consum_change
        vis_major = season_change + storage_change - pre_storage_consum_change
        return anthro_press, vis_major, anthro_press + vis_major
    
    def decomposition_p_dynamic_fail_v2(self, date_series, consum_series, pre_storage_series):
        # we preasent the order of the decomposition, t -> ws -> c 
        pf = self.risk_map_fail
        date_series = date_series.reset_index(drop = True)
        init_c_i = np.argmin(abs(self.consum_index_standard - consum_series[0]))
        fin_c_i = np.argmin(abs(self.consum_index_standard - consum_series[-1]))
        init_s_i = np.argmin(abs(self.consum_index_standard - pre_storage_series[0]))
        fin_s_i = np.argmin(abs(self.storage_index_standard - pre_storage_series[-1]))
        init_s_c_i = np.argmin(abs(self.storage_index_standard - (pre_storage_series[0] - consum_series[0])))
        init_t_i = date_series[0].month - 1
        fin_t_i = date_series[len(date_series)-1].month - 1 
        season_change = pf[fin_t_i][init_s_i][init_c_i] - pf[init_t_i][init_s_i][init_c_i]
        storage_change = pf[fin_t_i][fin_s_i][init_c_i] - pf[fin_t_i][init_s_i][init_c_i]
        pre_storage_consum_change = pf[fin_t_i][init_s_i][init_c_i] - pf[fin_t_i][init_s_c_i][init_c_i]
        consum_change = pf[fin_t_i][fin_s_i][fin_c_i] - pf[fin_t_i][fin_s_i][init_c_i]
        
        anthro_press = consum_change + pre_storage_consum_change
        vis_major = season_change + storage_change - pre_storage_consum_change
        return anthro_press, vis_major, anthro_press + vis_major

    def decision_support(self, month, initial_water_storage, pre_status,
                         pf_max = 0.05, 
                         ps_min = 0.95):
        agent_s_f = np.argmin(np.abs(cp.risk_map_cwdi - pf_max), axis = 2)
        agent_f_s = np.argmin(np.abs(cp.risk_map_sucess - ps_min), axis = 2)
        s_i = np.argmin(abs(initial_water_storage - cp.storage_index_standard))
        if pre_status == 0:
            c_i = agent_s_f[month - 1][s_i]
        else:
            c_i = agent_f_s[month - 1][s_i]
        plan_water_supply = cp.consum_index_standard[c_i]
        return plan_water_supply

    def water_management_policy(self, start, end, capacity = 3000):
        initial_storage = self.df_r["pre_storage"][start]
        initial_status = self.df_r["drought_record"][start - 1]
        inflow_array = np.array(self.df_r["inflow"])[start: end]
        correct_array = np.array(self.df_r["storage_correct"])[start:end]
        discharge_array = np.array(self.df_r["discharge_power"])[start: end]
        date_array = self.df_r["date"][start:end].reset_index(drop = True)
        threshold_array = np.array(self.df_r["threshold"])[start: end]
        bier_array = np.array(self.df_r["bier_l"])[start: end]
        ev_array = np.array(self.df_r["ev"])[start: end]

        storage_series = []
        rfd_series = []
        cwdi_series = []
        status_series = []
        consum_series = []
        for i in range(len(date_array)):
            inflow = inflow_array[i]
            correct = correct_array[i]
            discharge = discharge_array[i]
            month = date_array[i].month
            ev = ev_array[i]
            if i == 0:
                storage = initial_storage
                status = initial_status
                consum = self.decision_support(month, storage, status)
                consum_series.append(consum)
                next_storage = max(min(storage + inflow - consum - discharge + correct, capacity), 0)
                storage_state = (next_storage + storage) / 2 
                cta = consum / (inflow + storage_state)
                cwdi = np.reciprocal((1 + 99 * np.exp(-1* 19.85 * cta)))
                rfd = (consum - bier_array[i] * ev) * cwdi
                storage_series.append(storage)
                storage = next_storage
                if initial_status == 0:
                    if rfd > threshold_array[i] and cwdi > 0.85:
                        status_series.append(1)
                        status = 1
                    else:
                        status_series.append(0)
                        status = 0
                else:
                    if rfd <= threshold_array[i] and cwdi <= 0.85:
                        status_series.append(0)
                        status = 0
                    else:
                        status_series.append(1)
                        status = 1
                rfd_series.append(rfd)
                cwdi_series.append(cwdi)
            else:
                storage = storage
                status = status
                consum = self.decision_support(month, storage, status)
                consum_series.append(consum)
                next_storage = max(min(storage + inflow - consum - discharge + correct, capacity), 0)
                storage_state = (next_storage + storage) / 2
                cta = consum / (inflow + storage_state)
                cwdi = np.reciprocal((1 + 99 * np.exp(-1* 19.85 * cta)))
                rfd = (consum - bier_array[i] * ev) * cwdi
                storage_series.append(storage)
                storage = next_storage
                if status == 0:
                    if rfd > threshold_array[i] and cwdi > 0.85:
                        status_series.append(1)
                        status = 1
                    else:
                        status_series.append(0)
                        status = 0
                else:
                    if rfd <= threshold_array[i] and cwdi <= 0.85:
                        status_series.append(0)
                        status = 0
                    else:
                        status_series.append(1)
                        status = 1
                rfd_series.append(rfd)
                cwdi_series.append(cwdi)
        
        return np.array(storage_series), np.array(status_series), np.array(rfd_series), np.array(cwdi_series), np.array(consum_series)

    def rfd_cwdi_method_model(self):
        df_r = self.df_r
        drought_record = np.array(df_r["drought_record"])
        rfd_series = np.array(df_r["rfd"])
        cwdi_series = np.array(df_r["cwdi"])
        threshold_series = np.array(df_r["threshold"])
        output = [drought_record[0]]
        for i in range(1, len(rfd_series)):
            if output[i-1] == 0:
                if rfd_series[i] >= threshold_series[i] and cwdi_series[i] >= 0.85:
                    output.append(1)
                else:
                    output.append(0)
            else:
                if rfd_series[i] < threshold_series[i] and cwdi_series[i] < 0.85:
                    output.append(0)
                else:
                    output.append(1)
        return np.array(output)

def collect_drought_timing(df_r):
    record = np.array(df_r["drought_record"])
    output_start = []
    output_end = []
    for i in range(1, len(record)):
        if record[i] != record[i-1]:
            if record[i] == 1:
                output_start.append(i)
            elif record[i] == 0:
                output_end.append(i)
        else:
            continue
    return np.array(output_start), np.array(output_end)

def collect_drought_timing_2(status_series):
    record = status_series
    output_start = []
    output_end = []
    for i in range(1, len(record)):
        if record[i] != record[i-1]:
            if record[i] == 1:
                output_start.append(i)
            elif record[i] == 0:
                output_end.append(i)
        else:
            continue
    return np.array(output_start), np.array(output_end)

def event_irresistibility_analysis_fail(timing_index, timing = 1):
    vis_major_list_f = []
    anthro_press_list_f =[]
    pf_list = []
    for i in range(len(timing_index)):
        f_start = timing_index[i]
        pre_f_start = f_start - timing
        anthro_press, vis_major, dp = cp.decomposition_p_dynamic_fail_v1(pre_f_start, f_start)
        anthro_press_list_f.append(anthro_press)
        vis_major_list_f.append(vis_major)
        pf_list.append(dp)
        
    return np.array(vis_major_list_f), np.array(anthro_press_list_f)

def event_irresistibility_analysis_sucess(timing_index, timing = 1):
    vis_major_list_f = []
    anthro_press_list_f =[]
    pf_list = []
    for i in range(len(timing_index)):
        f_start = timing_index[i]
        pre_f_start = f_start - timing
        anthro_press, vis_major, dp = cp.decomposition_p_dynamic_sucess_v1(pre_f_start, f_start)
        anthro_press_list_f.append(anthro_press)
        vis_major_list_f.append(vis_major)
        pf_list.append(dp)
        
    return np.array(vis_major_list_f), np.array(anthro_press_list_f)

def event_irresistibility_analysis_fail_v2(timing_index, timing = 1):
    vis_major_list_f = []
    anthro_press_list_f =[]
    status_list = []
    for i in range(len(timing_index)):
        f_start = timing_index[i]
        pre_f_start = f_start - timing
        temp_output = cp.water_management_policy(pre_f_start, f_start)
        pre_storage_series = temp_output[0]
        consum_series = temp_output[4]
        date_series = df_r["date"][pre_f_start: f_start]
        anthro_press, vis_major, dp = cp.decomposition_p_dynamic_fail_v2(date_series, consum_series, pre_storage_series)
        anthro_press_list_f.append(anthro_press)
        vis_major_list_f.append(vis_major)
        status_list.append(temp_output[1])
        
    return np.array(vis_major_list_f), np.array(anthro_press_list_f), np.array(status_list)

def event_irresistibility_analysis_sucess_v2(timing_index, timing = 1):
    vis_major_list_f = []
    anthro_press_list_f =[]
    status_list = []
    for i in range(len(timing_index)):
        f_start = timing_index[i]
        pre_f_start = f_start - timing
        temp_output = cp.water_management_policy(pre_f_start, f_start)
        pre_storage_series = temp_output[0]
        consum_series = temp_output[4]
        date_series = df_r["date"][pre_f_start: f_start]        
        anthro_press, vis_major, dp = cp.decomposition_p_dynamic_sucess_v2(date_series, consum_series, pre_storage_series)
        anthro_press_list_f.append(anthro_press)
        vis_major_list_f.append(vis_major)
        status_list.append(temp_output[1])
        
    return np.array(vis_major_list_f), np.array(anthro_press_list_f), np.array(status_list)

def draw_year_example(date, 
                      cwdi,
                      rfd, 
                      threshold,
                      ps,
                      pf,
                      start_index,
                      end_index,
                      sci, 
                      ssi, 
                      sii,
                      start,
                      end):
        
    date_series = date[start:end]
    cwdi_series = cwdi[start:end]
    rfd_series = rfd[start:end]
    threshold_series = threshold[start:end] 
    sci_series = sci[start:end]
    ssi_series = ssi[start:end]
    sii_series = sii[start:end]
    ps_series = ps[start:end]
    pf_series = pf[start:end]
    cwdi_a = np.array(cwdi)
    cwdi_a.sort()

    cwdi_75 = cwdi_a[int(len(cwdi_a) * 0.75)]
    fig = plt.figure(dpi =600, figsize = (10, 8))
    spec = gridspec.GridSpec( nrows = 5, ncols = 1)
    
    fig.add_subplot(spec[0])
    plt.bar(date_series, (threshold_series - rfd_series)/ threshold_series,
             color = "r", width = 5, label = "\u0394RFD / Threshold")
    plt.axhline(0, color = "black", alpha = 0.5, linewidth = 0.5)
    plt.ylabel("ratio")
    plt.legend(loc = "upper left", bbox_to_anchor = (0.75,0.35))
    plt.ylim((-1.5,1.5))
    plt.xticks([])
    
    fig.add_subplot(spec[1])
    plt.plot(date_series, cwdi_series, color = "orange", label = "CWDI")
    plt.axhline(cwdi_75, color = "purple", linestyle = "--", label = "recovery criteria")
    plt.xticks([])
    plt.ylabel("CWDI")
    plt.ylim((-0.1,1.1))
    plt.legend(loc = "upper left", bbox_to_anchor = (0.4,0.35), ncol = 2)
    
    fig.add_subplot(spec[2])
    plt.plot(date_series, sci_series, color = "brown", label = "C*")
    plt.plot(date_series, ssi_series, color = "green", label = "WS*")
    plt.plot(date_series, sii_series, color = "blue", label = "A*")
    plt.xticks([])
    plt.ylabel("Standardized quantity")
    plt.legend(loc = "upper left", bbox_to_anchor = (0.4,1), ncol = 3)
    for i in range(len(start_index)):
        event_start = start_index[i]
        event_end = end_index[i]
        date_record = np.arange(start, end)
        if event_start in date_record and event_end in date_record:
            plt.axvspan(date_series[event_start],
                         date_series[event_end],
                         color = "red", alpha = 0.2)
        else:
            continue
      
    fig.add_subplot(spec[3])
    plt.bar(date_series, ps_series * 100, color = "blue", width = 5, label = "P$_{S}$")
    plt.ylabel("probability %")
    plt.ylim(-10, 110)
    plt.xticks([])
    plt.legend()

    fig.add_subplot(spec[4])
    plt.bar(date_series, pf_series * 100, color = "purple", width = 5, label = "P$_{F}$")
    plt.ylabel("probability %")
    plt.ylim(-10, 110)
    plt.xticks(rotation = -15)
    plt.legend()
    plt.show()

def get_contour_diagram_subplot(riskmap_sucess,
                                riskmap_fail,
                                riskmap_rfd,
                                riskmap_cwdi,
                                date_series,
                                consumption_series,
                                storage_series,
                                record_series):
    
    c_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
    s_list = [[],[],[],[],[],[],[],[],[],[],[],[]]
    unit = 86400 / 1000000
    s_standard = cp.storage_index_standard * unit
    c_standard = cp.consum_index_standard * unit
    consumption_series = consumption_series * unit
    storage_series = storage_series * unit
    c_list_s = [[],[],[],[],[],[],[],[],[],[],[],[]]
    s_list_s = [[],[],[],[],[],[],[],[],[],[],[],[]]
    c_list_f = [[],[],[],[],[],[],[],[],[],[],[],[]]
    s_list_f = [[],[],[],[],[],[],[],[],[],[],[],[]]
    
    for i in range(1, len(date_series)):
        consum =  c_standard[np.argmin(abs(consumption_series[i] - c_standard))]
        storage = s_standard[np.argmin(abs(storage_series[i] - s_standard))]
        if record_series[i - 1] == 0:
            for j in range(12):            
                if j + 1 == date_series[i].month:
                    c_list_s[j].append(consum)
                    s_list_s[j].append(storage)
        else:
            for j in range(12):            
                if j + 1 == date_series[i].month:
                    c_list_f[j].append(consum)
                    s_list_f[j].append(storage)
        for j in range(12):            
            if j + 1 == date_series[i].month:
                c_list[j].append(consum)
                s_list[j].append(storage)
    
    for i in range(len(c_list)):
        c_list[i] = np.array(c_list[i])
        s_list[i] = np.array(s_list[i])
        c_list_s[i] = np.array(c_list_s[i])
        s_list_s[i] = np.array(s_list_s[i])
        c_list_f[i] = np.array(c_list_f[i])
        s_list_f[i] = np.array(s_list_f[i])

    leng = len(riskmap_fail)
    fig = plt.figure(dpi = 600, figsize = (14,9)) 
    spec = gridspec.GridSpec( nrows = 3, ncols = 4)
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for i in range(leng):
        fig.add_subplot(spec[i])
        plt.title(month[i])
        fail = riskmap_fail[i]
        C, S = np.meshgrid(c_standard, s_standard, indexing = "xy")
        plt.contourf(C,S, fail, 10, cmap = cm.get_cmap("Purples"),alpha = 0.8)
        plt.scatter(c_list_s[i], s_list_s[i], c = "black", marker = "v",s = 50
                    , label = "Normal period")

        if (i <= 3 and i >= 1 ) or (i >= 5 and i <= 7)  :
            plt.xticks(color = "w")
            plt.yticks(color = "w")
        elif (i+1) % 4 == 1 and i != 8:
            plt.xticks(color = "w")
            plt.ylabel("storage (M$m^3$)")
        elif i > 8:
            plt.yticks(color = "w")
            plt.xlabel("consumption (M$m^3$)")       
            
        elif i == 8:
            plt.ylabel("storage (M$m^3$)")
            plt.xlabel("consumption (M$m^3$)")
            plt.legend()
        
    cbar_ax = fig.add_axes([0.92,0.1,0.01,0.8]) 
    c_bar = fig.colorbar(plt.contourf(C,S, fail, 10, cmap = cm.get_cmap("Purples")),
                        cax = cbar_ax, label = "P$_{F}$")
    c_bar.set_label("P$_{F}$", fontsize = 18)
    c_bar.ax.tick_params(labelsize = 14)
    plt.show()

    for i in range(len(c_list)):
        c_list[i] = np.array(c_list[i])
        s_list[i] = np.array(s_list[i])

    leng = len(riskmap_sucess)
    fig = plt.figure(dpi = 600, figsize = (14,9)) 
    spec = gridspec.GridSpec( nrows = 3, ncols = 4)
    month = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for i in range(leng):
        fig.add_subplot(spec[i])
        plt.title(month[i])
        sucess = riskmap_sucess[i]
        C, S = np.meshgrid(c_standard, s_standard, indexing = "xy")
        plt.contourf(C,S, sucess, 10, cmap = cm.get_cmap("Blues"), alpha = 0.8)
        plt.scatter(c_list_f[i], s_list_f[i], c = "black", marker = "*",s = 70
                    , label = "Water restriction period")

        if (i <= 3 and i >= 1 ) or (i >= 5 and i <= 7)  :
            plt.xticks(color = "w")
            plt.yticks(color = "w")
        elif (i+1) % 4 == 1 and i != 8:
            plt.xticks(color = "w")
            plt.ylabel("storage (M$m^3$)")
        elif i > 8:
            plt.yticks(color = "w")
            plt.xlabel("consumption (M$m^3$)")       
            
        elif i == 8:
            plt.ylabel("storage (M$m^3$)")
            plt.xlabel("consumption (M$m^3$)")        
            plt.legend()

    cbar_ax = fig.add_axes([0.92,0.1,0.01,0.8]) 
    c_bar = fig.colorbar(plt.contourf(C,S, sucess, 10, cmap = cm.get_cmap("Blues"), alpha = 0.8),
                 cax = cbar_ax, label = "P$_{S}$")
    c_bar.set_label("P$_{S}$", fontsize = 18)
    c_bar.ax.tick_params(labelsize = 14)
    plt.show()

def check_crps(df_r):
    pf_series = []
    ps_series = []
    p_sample = []
    p_all_series = []
    pf_sample = []
    ps_sample = []
    rfd_cwdi_record = cp.rfd_cwdi_method_model()
    for i in range(len(df_r)):
        s_i = np.argmin(abs(df_r["pre_storage"][i] - cp.storage_index_standard))
        c_i = np.argmin(abs(df_r["consumption"][i] - cp.consum_index_standard))
        month = df_r["date"][i].month
        pf_series.append(cp.risk_map_fail[int(month - 1)][s_i][c_i])
        ps_series.append(cp.risk_map_sucess[int(month - 1)][s_i][c_i])
# =============================================================================
#         if i == 0:
#             p_all_series.append(cp.risk_map_fail[int(month - 1)][s_i][c_i])
#         elif df_r["drought_record"][i-1] == 0:
#             p_all_series.append(cp.risk_map_fail[int(month - 1)][s_i][c_i])
#         elif df_r["drought_record"][i-1] == 1:
#             p_all_series.append(1 - cp.risk_map_sucess[int(month - 1)][s_i][c_i])
#         else:
#             p_all_series.append(cp.risk_map_fail[int(month - 1)][s_i][c_i])
# =============================================================================
        if i == 0:
            p_all_series.append(cp.risk_map_fail[int(month - 1)][s_i][c_i])
        elif rfd_cwdi_record[i-1] == 0:
            p_all_series.append(cp.risk_map_fail[int(month - 1)][s_i][c_i])
        elif rfd_cwdi_record[i-1] == 1:
            p_all_series.append(1 - cp.risk_map_sucess[int(month - 1)][s_i][c_i])
        else:
            p_all_series.append(cp.risk_map_fail[int(month - 1)][s_i][c_i])

    pf_series = np.array(pf_series)
    ps_series = np.array(ps_series)
    p_all_series = np.array(p_all_series)
    bin_series = st.bernoulli.rvs(0.5, size = len(df_r))
    drought_record = np.array(df_r["drought_record"])
    average_record = st.bernoulli.rvs(np.mean(drought_record), size = len(df_r))
    for i in range(len(p_all_series)):
        p_sample.append(st.bernoulli.rvs(p_all_series[i], size = 1))
        pf_sample.append(st.bernoulli.rvs(pf_series[i], size = 1))
        ps_sample.append(st.bernoulli.rvs(1- ps_series[i], size = 1))
    p_sample = np.array(p_sample)
    pf_sample = np.array(pf_sample)
    ps_sample = np.array(ps_sample)
    return p_sample ,bin_series, average_record, rfd_cwdi_record, pf_sample, ps_sample, p_all_series
    
def crps_1(y_true, y_pred, sample_weight=None):
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)
    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)
    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, -1)
    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    return np.average(per_obs_crps, weights=sample_weight)

df_r = remake_df(df)
ssi = standardized_process(df_r, "pre_storage").transformation_v2()
sci = standardized_process(df_r, "consumption").transformation_v2()
sii = standardized_process(df_r, "inflow").transformation_v2()

cp = conditional_probability_analysis(df_r)
start_index, end_index = collect_drought_timing(df_r)
pf_series, ps_series = cp.transform_probability_dynamics()

vis_occur_f, lamda_occur_f = event_irresistibility_analysis_fail(start_index, 1)
vis_termin_s, lamda_termin_s = event_irresistibility_analysis_sucess(end_index, 1)

vis_occur_r_f, lamda_occur_r_f, status_list_r_occur = event_irresistibility_analysis_fail_v2(start_index, 1)
vis_termin_r_s, lamda_termin_r_s, status_list_r_termin = event_irresistibility_analysis_sucess_v2(start_index + 1, 1)
vis_termin_b, lamda_termin_b = event_irresistibility_analysis_sucess(start_index + 1, 1)
draw_year_example(df_r["date"], df_r["cwdi"], df_r["rfd"], 
                  df_r["threshold"], ps_series, pf_series,
                  start_index, end_index,
                  sci, ssi, sii, 230, 280)

get_contour_diagram_subplot(cp.risk_map_sucess,
                            cp.risk_map_fail,
                            cp.risk_map_rfd, 
                            cp.risk_map_cwdi,
                            df_r["date"],
                            df_r["consumption"],
                            df_r["pre_storage"],
                            df_r["drought_record"])

p_sample ,bin_series, average_record, drought_record, pf_sample, ps_sample, p_all_series = check_crps(df_r)
# =============================================================================
# p_crps_1 = crps_1(np.array(df_r["drought_record"]), p_all_series)
# bin_crps_1 = crps_1(np.array(df_r["drought_record"]), 0.5 * np.ones(shape = len(drought_record)))
# p_mean_crps_1 = crps_1(np.array(df_r["drought_record"]), np.mean(drought_record) * np.ones(shape = len(drought_record)))
# p_s_crps_1 = crps_1(drought_record, 1 - ps_series)
# p_f_crps_1 = crps_1(drought_record, pf_series)
# =============================================================================

def decision_making_comparison(df_r, start_index, 
                               end_index, previous_time = 1):
    status_r_record_start = []
    status_r_record_end = []
    for i in range(len(start_index)):
        start_s = start_index[i] - previous_time
        end_s = start_index[i]
        temp_output = cp.water_management_policy(start_s, end_s)
        status_r_record_start.append(temp_output[1])
        
        start_e = end_index[i] - previous_time
        end_e = end_index[i] - previous_time
        temp_output = cp.water_management_policy(start_e, end_e)
        status_r_record_end.append(temp_output[1])
    
    vis_occur_f_r, lamda_occur_f_r = event_irresistibility_analysis_fail_v2(start_index, 1)
    vis_termin_s_r, lamda_termin_s_r = event_irresistibility_analysis_sucess_v2(end_index, 1)
    
    return status_r_record_start, status_r_record_end

def get_bin_array(p_series, drought_record):
    
    bool_series_f = p_series >= 0.5
    int_series_f = bool_series_f.astype(int)
    s_t_s = 0
    f_t_f = 0
    s_t_f = 0
    f_t_s = 0
    for i in range(len(p_series)):
        if int_series_f[i] == 0:
            if int_series_f[i] == drought_record[i]:
                s_t_s += 1
            else:
                s_t_f += 1
        else:
            if int_series_f[i] == drought_record[i]:
                f_t_f += 1
            else:
                f_t_s += 1
    return s_t_s, s_t_f, f_t_s, f_t_f
    
def plot_Lam_G_pair(Lam_F, G_F, Lam_S, G_S, *,
                    labels=('F', 'S'),
                    markers=('v', '*'),
                    panel_titles=('(a)', '(b)'),
                    point_colors=('black', 'black'),   # NEW
                    dpi=600, figsize_cm=(15, 10),
                    fs_axis=8, fs_title=10, fs_ratio=7,
                    shade_alpha=0.15):

    def _add_type_ratio(ax, r):
        spec = [("Type I",  1.95,  0, 'right',  'center'),
                ("Type II", 0,  1.95, 'center', 'top'),
                ("Type III",-1.95,  0, 'left',   'center'),
                ("Type IV", 0, -1.95, 'center', 'bottom')]
        for (txt, x, y, ha, va), p in zip(spec, r):
            ax.text(x, y, f'{txt}\n{p*100:4.1f} %', ha=ha, va=va,
                    fontsize=fs_ratio)
            
    def _scatter_panel(ax, Lam, G, marker, color, sub):
        Lam, G = np.asarray(Lam), np.asarray(G)
        dp = Lam + G
        counts = np.array([((dp>=0)&(Lam> G)).sum(),
                           ((dp> 0)&(Lam<=G)).sum(),
                           ((dp<=0)&(Lam< G)).sum(),
                           ((dp< 0)&(Lam>=G)).sum()])
        ratios = counts / counts.sum()

        # background shading
        X,Y = np.meshgrid(np.linspace(-2,2,401), np.linspace(-2,2,401))
        mask = np.abs(X) > np.abs(Y)
        cmap = ListedColormap([[1,0,0,shade_alpha],[1,1,1,0]])
        ax.imshow(mask.astype(int), origin='lower',
                  extent=[-2,2,-2,2], cmap=cmap, interpolation='nearest')

        ax.plot([-2.5,2.5],[2.5,-2.5],'k--',lw=1)
        ax.text(-1.3,1.5,fr'$\Delta P_{{{sub}}}=0$',rotation=-45,fontsize=8,
                ha='center',va='center')

        # scatter with chosen colour
        ax.plot(Lam, G, marker, ms=5,
                markerfacecolor=color, markeredgecolor=color)

        _add_type_ratio(ax, ratios)
        ax.set_xlim(-2,2); ax.set_ylim(-2,2); ax.set_aspect('equal')
        ax.tick_params(labelsize=fs_axis)
        ax.set_xlabel(fr'$\lambda_{{{sub}}}$', fontsize=fs_axis)
        ax.set_ylabel(fr'$G_{{{sub}}}$',        fontsize=fs_axis)

    cm2in=1/2.54
    fig,axs=plt.subplots(1,2,dpi=dpi,
                         figsize=(figsize_cm[0]*cm2in,
                                  figsize_cm[1]*cm2in))

    _scatter_panel(axs[0], Lam_F, G_F, markers[0], point_colors[0], labels[0])
    axs[0].set_title(panel_titles[0], fontsize=fs_title)

    _scatter_panel(axs[1], Lam_S, G_S, markers[1], point_colors[1], labels[1])
    axs[1].set_title(panel_titles[1], fontsize=fs_title)

    plt.tight_layout(w_pad=0.5)
    plt.show()
    return fig, axs

def plot_Lam_G_pair_dis_s(lam_f, g_f, lam_s, g_s, *,
                          labels=('S', 'S'),
                          markers=('*', '*'),
                          panel_titles=('(a)', '(b)'),
                          alt_labels=(None, (r'$\Delta\lambda_S$', r'$\Delta G_S$')),
                          point_colors=('blue', 'blue'),  # ← updated colors here
                          dpi=600, figsize_cm=(15, 10),
                          fs_axis=8, fs_title=10, fs_ratio=7,
                          shade_alpha=0.15):

    def _ratio_text(ax, r):
        spec = [("Type I", 1.95, 0, 'right', 'center'),
                ("Type II", 0, 1.95, 'center', 'top'),
                ("Type III", -1.95, 0, 'left', 'center'),
                ("Type IV", 0, -1.95, 'center', 'bottom')]
        for (txt, x, y, ha, va), p in zip(spec, r):
            ax.text(x, y, f'{txt}\n{p*100:4.1f} %', ha=ha, va=va, fontsize=fs_ratio)

    def _panel(ax, lam, g, marker, color, xlab, ylab, side,
               show_types=True, add_legend=False, legend_label=""):
        lam, g = np.asarray(lam), np.asarray(g)
        dp = lam + g
        counts = np.array([
            ((dp >= 0) & (lam > g)).sum(),
            ((dp >  0) & (lam <= g)).sum(),
            ((dp <= 0) & (lam <  g)).sum(),
            ((dp <  0) & (lam >= g)).sum()])
        ratios = counts / counts.sum()

        X, Y = np.meshgrid(np.linspace(-2, 2, 401), np.linspace(-2, 2, 401))
        mask = np.abs(X) > np.abs(Y)
        cmap = ListedColormap([[1, 0, 0, shade_alpha], [1, 1, 1, 0]])
        ax.imshow(mask.astype(int), origin='lower',
                  extent=[-2, 2, -2, 2], cmap=cmap, interpolation='nearest')

        ax.plot([-2.5, 2.5], [2.5, -2.5], 'k--', lw=1)
        diag_label = r'$\Delta(\Delta P_S)=0$' if side == 1 else r'$\Delta P_S=0$'
        ax.text(-1.3, 1.5, diag_label, rotation=-45, fontsize=8,
                ha='center', va='center')

        # Colored scatter points
        ax.plot(lam, g, marker, ms=5,
                markerfacecolor=color, markeredgecolor=color,
                label=legend_label if add_legend else None)

        if show_types:
            _ratio_text(ax, ratios)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=fs_axis)
        ax.set_xlabel(xlab, fontsize=fs_axis)
        ax.set_ylabel(ylab, fontsize=fs_axis)

        if add_legend:
            ax.legend(loc='lower center', fontsize=fs_axis, frameon=False)

    cm2in = 1 / 2.54
    fig, axs = plt.subplots(1, 2, dpi=dpi,
                            figsize=(figsize_cm[0] * cm2in,
                                     figsize_cm[1] * cm2in))

    x0, y0 = fr'$\lambda_{{{labels[0]}}}$', fr'$G_{{{labels[0]}}}$'
    x1, y1 = fr'$\lambda_{{{labels[1]}}}$', fr'$G_{{{labels[1]}}}$'
    if alt_labels[0]: x0, y0 = alt_labels[0]
    if alt_labels[1]: x1, y1 = alt_labels[1]

    _panel(axs[0], lam_f, g_f, markers[0], point_colors[0], x0, y0, side=0,
           show_types= False, add_legend=True,
           legend_label="Success in acceleration")
    axs[0].set_title(panel_titles[0], fontsize=fs_title)

    _panel(axs[1], lam_s, g_s, markers[1], point_colors[1], x1, y1, side=1,
           show_types=False, add_legend=False)
    axs[1].set_title(panel_titles[1], fontsize=fs_title)

    plt.tight_layout(w_pad=0.5)
    plt.show()
    return fig, axs

def plot_Lam_G_pair_dis_f(lam_f, g_f, lam_s, g_s, *,
                          labels=('F', 'F'),
                          markers=('v', 'v'),
                          panel_titles=('(a)', '(b)'),
                          alt_labels=(None, (r'$\Delta\lambda_F$', r'$\Delta G_F$')),
                          point_colors=('purple', 'purple'),  # ← updated colors here
                          dpi=600, figsize_cm=(15, 10),
                          fs_axis=8, fs_title=10, fs_ratio=7,
                          shade_alpha=0.15):

    def _ratio_text(ax, r):
        spec = [("Type I", 1.95, 0, 'right', 'center'),
                ("Type II", 0, 1.95, 'center', 'top'),
                ("Type III", -1.95, 0, 'left', 'center'),
                ("Type IV", 0, -1.95, 'center', 'bottom')]
        for (txt, x, y, ha, va), p in zip(spec, r):
            ax.text(x, y, f'{txt}\n{p*100:4.1f} %', ha=ha, va=va, fontsize=fs_ratio)

    def _panel(ax, lam, g, marker, color, xlab, ylab, side,
               show_types=True, add_legend=False, legend_label=""):
        lam, g = np.asarray(lam), np.asarray(g)
        dp = lam + g
        counts = np.array([
            ((dp >= 0) & (lam > g)).sum(),
            ((dp >  0) & (lam <= g)).sum(),
            ((dp <= 0) & (lam <  g)).sum(),
            ((dp <  0) & (lam >= g)).sum()])
        ratios = counts / counts.sum()

        X, Y = np.meshgrid(np.linspace(-2, 2, 401), np.linspace(-2, 2, 401))
        mask = np.abs(X) > np.abs(Y)
        cmap = ListedColormap([[1, 0, 0, shade_alpha], [1, 1, 1, 0]])
        ax.imshow(mask.astype(int), origin='lower',
                  extent=[-2, 2, -2, 2], cmap=cmap, interpolation='nearest')

        ax.plot([-2.5, 2.5], [2.5, -2.5], 'k--', lw=1)
        diag_label = r'$\Delta(\Delta P_F)=0$' if side == 1 else r'$\Delta P_F=0$'
        ax.text(-1.3, 1.5, diag_label, rotation=-45, fontsize=8,
                ha='center', va='center')

        # Colored scatter points
        ax.plot(lam, g, marker, ms=5,
                markerfacecolor=color, markeredgecolor=color,
                label=legend_label if add_legend else None)

        if show_types:
            _ratio_text(ax, ratios)

        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=fs_axis)
        ax.set_xlabel(xlab, fontsize=fs_axis)
        ax.set_ylabel(ylab, fontsize=fs_axis)

        if add_legend:
            ax.legend(loc='lower center', fontsize=fs_axis, frameon=False)

    cm2in = 1 / 2.54
    fig, axs = plt.subplots(1, 2, dpi=dpi,
                            figsize=(figsize_cm[0] * cm2in,
                                     figsize_cm[1] * cm2in))

    x0, y0 = fr'$\lambda_{{{labels[0]}}}$', fr'$G_{{{labels[0]}}}$'
    x1, y1 = fr'$\lambda_{{{labels[1]}}}$', fr'$G_{{{labels[1]}}}$'
    if alt_labels[0]: x0, y0 = alt_labels[0]
    if alt_labels[1]: x1, y1 = alt_labels[1]

    _panel(axs[0], lam_f, g_f, markers[0], point_colors[0], x0, y0, side=0,
           show_types= False, add_legend=True,
           legend_label="Success in prevention")
    axs[0].set_title(panel_titles[0], fontsize=fs_title)

    _panel(axs[1], lam_s, g_s, markers[1], point_colors[1], x1, y1, side=1,
           show_types=False, add_legend=False)
    axs[1].set_title(panel_titles[1], fontsize=fs_title)

    plt.tight_layout(w_pad=0.5)
    plt.show()
    return fig, axs

def draw_event_duration_cdf(duration_series, duration_r_series,
                             *,
                             labels=('Historical record', 'Policy-guided'),
                             xlabel='Duration (months)',
                             ylabel='ECDF',
                             figsize=(6, 4),
                             fontsize=12,
                             dpi=600):
    """
    Draws CDF plot comparing two duration series.
    Filters out durations < 1 month and clamps x-axis to [0, 12].

    Parameters
    ----------
    duration_series : array-like
        Durations from historical events.
    duration_r_series : array-like
        Durations from policy-guided events.
    labels : tuple of str
        Labels for the two datasets.
    xlabel, ylabel, title : str
        Axis labels and plot title.
    figsize : tuple
        Figure size in inches.
    fontsize : int
        Font size for labels.
    dpi : int
        Output resolution.
    """
    # Filter durations ≥ 1 month
    d1 = np.array(duration_series)
    d2 = np.array(duration_r_series)
    d1 = d1[d1 >= 1]
    d2 = d2[d2 >= 1]

    # Create distributions
    d1_dist = st.rv_histogram(np.histogram(d1, bins=len(d1) - 1, range=(1, 12)))
    d2_dist = st.rv_histogram(np.histogram(d2, bins=len(d2) - 1, range=(1, 12)))

    x_vals = np.linspace(0, 12, 500)

    # Plot
    plt.figure(figsize=figsize, dpi=dpi)
    plt.plot(x_vals, d1_dist.cdf(x_vals), label=f"{labels[0]} ({len(d1)} events)", color="blue")
    plt.plot(x_vals, d2_dist.cdf(x_vals), label=f"{labels[1]} ({len(d2)} events)", color="orange")

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.ylim(0, 1)
    plt.xlim(0, 12)
    plt.legend(fontsize=fontsize)
    plt.tight_layout()
    plt.show()


plot_Lam_G_pair(lamda_occur_f, vis_occur_f, lamda_termin_s, vis_termin_s)

plot_Lam_G_pair_dis_s(lamda_termin_r_s, vis_termin_r_s, lamda_termin_r_s - lamda_termin_b
                , vis_termin_r_s - vis_termin_b)

plot_Lam_G_pair_dis_f(lamda_occur_r_f, vis_occur_r_f, lamda_occur_r_f - lamda_occur_f
                , vis_occur_r_f - vis_occur_f)

duration_r = collect_drought_timing_2(cp.water_management_policy(1, len(df_r))[1])[1] - collect_drought_timing_2(cp.water_management_policy(1, len(df_r))[1])[0]
duration = collect_drought_timing(df_r)[1] - collect_drought_timing(df_r)[0]
