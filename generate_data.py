import numpy as np

from random import seed
import random
seed(213)

def get_substellar_longitude(t, initial_substellar_longitude, w_rot):
    
    substellar_longitude = initial_substellar_longitude - (w_rot * t)
    
    while substellar_longitude < 0.:
        substellar_longitude = substellar_longitude + (2 * np.pi)
        
    while substellar_longitude > (2. * np.pi):
        substellar_longitude = substellar_longitude - (2 * np.pi)
        
    return substellar_longitude

def get_albedo_sum_term(albedo_slice_i, substellar_longitude, longitudinal_boundary_angles_i_i1):
    
    phi_i = longitudinal_boundary_angles_i_i1[0]
    phi_i1 = longitudinal_boundary_angles_i_i1[1]
    T_E = substellar_longitude + (np.pi / 2.)
    T_W = substellar_longitude - (np.pi / 2.)
    
    if T_E > (2 * np.pi):
        T_E = T_E - (2 * np.pi)
        
    if T_W < (2 * np.pi):
        T_W = T_W + (2 * np.pi)
    
    while T_E < T_W:
        T_W = T_W - (2 * np.pi)
        
    if phi_i > phi_i1:
        phi_i -= (2 * np.pi)
    
    max_value = phi_i
    min_value = phi_i1
    
    if T_W > phi_i:
        max_value = T_W
        
    if T_E < phi_i1:
        min_value = T_E
        
    #print("Eastern Terminator: " + str(T_E))
    #print("phi_i+1: " + str(phi_i1))
    #print("Min value: " + str(min_value))
    
    #print("Western Terminator: " + str(T_W))
    #print("phi_i: " + str(phi_i))
    #print("Max value: " + str(max_value))

    first_value = (min_value / 2.) + (np.sin(2 * (min_value - substellar_longitude)) / 4.)
    second_value = (max_value / 2.) + (np.sin(2 * (max_value - substellar_longitude)) / 4.)
    
    #print("First value: " + str(first_value))
    #print("Second value: " + str(second_value))
    
    albedo_sum_term = albedo_slice_i * (first_value - second_value)
    
    #print("Albedo Sum Term: " + str(albedo_sum_term))
    
    if albedo_sum_term <= 0.:
        returned_val = 0.
    
    else:
        returned_val = albedo_sum_term
        
    #print("Returned Value: " + str(returned_val))
        
    return returned_val

def generate_synthetic_albedo_lightcurve_data(number_Of_Slices = 4, number_Of_Times = 20, time_in_days_per_time_interval = 0.21, verbose = False):

    times = [time_in_days_per_time_interval * t for t in range(number_Of_Times)]

    m = number_Of_Slices

    longitudinal_boundary_angles = [(2. * np.pi) * (i / m) for i in range(m)]
    slices_of_albedo = [random.random() for i in range(m)]

    initial_substellar_longitude = 2. * np.pi * random.random()

    w_rot = (2. * np.pi) * (random.random() + 0.5)
    
    if verbose == True:
          
        print("Initial Substellar Longitude: " + str(initial_substellar_longitude))
        print("Rotational Angular Frequency: " + str(w_rot))
        print("Albedo Slice Values: ")
        print(slices_of_albedo)
    
    albedo_lightcurve_data_no_errors = []
    
    for t in times:
        
        all_albedo_sum_terms = []
        
        for i in range(m):
            
            substellar_longitude = get_substellar_longitude(t, initial_substellar_longitude, w_rot)
            
            phi_i = longitudinal_boundary_angles[i]
            phi_i1 = longitudinal_boundary_angles[(i + 1) % m]
            
            albedo_sum_term = get_albedo_sum_term(slices_of_albedo[i],\
                                           substellar_longitude,\
                                           (phi_i, phi_i1))
          
            all_albedo_sum_terms.append(albedo_sum_term)
            
        albedo_in_lightcurve = np.sum(np.array(all_albedo_sum_terms)) / np.pi
        
        albedo_lightcurve_data_no_errors.append(albedo_in_lightcurve)

    max_albedo_in_lightcurve = max(albedo_lightcurve_data_no_errors)
    albedo_error = max_albedo_in_lightcurve * 0.02

    albedo_lightcurve_data_with_errors = [-1. for albedo in albedo_lightcurve_data_no_errors]
    
    while not all(i >= 0. for i in albedo_lightcurve_data_with_errors):
        albedo_lightcurve_data_with_errors = [albedo + random.normalvariate(0., albedo_error) for albedo in albedo_lightcurve_data_no_errors]
        
    errors_on_data = [albedo_error for i in range(len(albedo_lightcurve_data_no_errors))]
    
    if verbose == True:
        print("Albedo Lightcurve Generated: " + str(albedo_lightcurve_data_with_errors))
    
    #init_values = {'w_rot': w_rot,\
                   #'initial_substellar_longitude': initial_substellar_longitude,\
                   #'slices_of_albedo': slices_of_albedo}
    
    init_values = {'w_rot': (2. * np.pi) * (random.random() + 0.5),\
                   'initial_substellar_longitude': 2. * np.pi * random.random(),\
                   'slices_of_albedo': [random.random() for i in range(m)]}
    
    return albedo_lightcurve_data_no_errors, albedo_lightcurve_data_with_errors,\
errors_on_data, times, longitudinal_boundary_angles, init_values