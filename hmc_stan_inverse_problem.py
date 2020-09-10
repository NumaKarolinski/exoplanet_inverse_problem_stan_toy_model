cores_to_use = 4

import os
os.environ['STAN_NUM_THREADS'] = str(cores_to_use)

import arviz
import numpy as np
import pystan
from matplotlib import pyplot as plt
from random import seed
import random
seed(213)

import generate_data

m = 3
num_times = 24

albedo_lightcurve_data_no_errors, albedo_lightcurve_data_with_errors, errors_on_data, times, longitudinal_boundary_angles, init_values = generate_data.generate_synthetic_albedo_lightcurve_data(number_Of_Slices = m, number_Of_Times = num_times, time_in_days_per_time_interval = (1. / 24.), verbose = True)

ocode = '''
functions{

    real get_substellar_longitude(real t, real initial_substellar_longitude, real w_rot){

        real substellar_longitude = initial_substellar_longitude - (w_rot * t);

        while (substellar_longitude < 0){
            substellar_longitude = substellar_longitude + (2 * pi());
        }

        while (substellar_longitude > (2 * pi())){
            substellar_longitude = substellar_longitude - (2 * pi());
        }
        
        return substellar_longitude;
    }
    
    real get_albedo_sum_term(real albedo_slice_i, real substellar_longitude, real phi_i, real phi_i1){

        real max_value;
        real min_value;
        
        real first_value;
        real second_value;
        
        real albedo_sum_term;
        
        real T_E = substellar_longitude + (pi() / 2);
        real T_W = substellar_longitude - (pi() / 2);

        if (T_E > (2 * pi())){
            T_E = T_E - (2 * pi());
        }

        if (T_W < (2 * pi())){
            T_W = T_W + (2 * pi());
        }
        
        while (T_E < T_W){
            T_W = T_W - (2 * pi());
        }
        
        if (phi_i > phi_i1){
            max_value = phi_i - (2 * pi());
        }
        
        else{
            max_value = phi_i;
        }

        min_value = phi_i1;

        if (T_W > phi_i){
            max_value = T_W;
        }

        if (T_E < phi_i1){
            min_value = T_E;
        }

        first_value = (min_value / 2) + (sin(2 * (min_value - substellar_longitude)) / 4);
        second_value = (max_value / 2) + (sin(2 * (max_value - substellar_longitude)) / 4);

        albedo_sum_term = albedo_slice_i * (first_value - second_value);
        
        //print("phi_i: ", phi_i);
        //print("Western Terminator: ", T_W);
        //print("Max value: ", max_value);
        //print("phi_i+1: ", phi_i1);
        //print("Eastern Terminator: ", T_E);
        //print("Min value: ", min_value);
        //print("First - Second: ", (first_value - second_value));
        //print("Albedo sum term: ", albedo_sum_term);
        
        if (albedo_sum_term <= 0){
            albedo_sum_term = 0;
        }

        return albedo_sum_term;
    }
    
    real forwardModel_lpdf(real[] lightcurve_data, real[] generated_lightcurve, real[] errors){
    
        int number_of_data_points = dims(generated_lightcurve)[1];
        real returnedValue = 0;
        
        for (t in 1:number_of_data_points){
            returnedValue -= 0.5 * ((((lightcurve_data[t] - generated_lightcurve[t])^2) / (errors[t]^2)));
        }
        
        //print("Returned Value: ", returnedValue);
        
        return returnedValue;
    }
}

data{

    int<lower = 1> number_of_data_points;
    int<lower = 1> number_of_slices;
    real<lower = 0, upper = 1> lightcurve_data[number_of_data_points];
    real<lower = 0> times[number_of_data_points];
    real<lower = 0, upper = 0.02> errors[number_of_data_points];
    real<lower = 0, upper = 2 * pi()> longitudinal_boundary_angles[number_of_slices];
}

transformed data{

    real phi_i[number_of_slices];
    real phi_i1[number_of_slices];
    
    for (i in 1:number_of_slices){
        
        phi_i[i] = longitudinal_boundary_angles[i];
        phi_i1[i] = longitudinal_boundary_angles[(i % number_of_slices) + 1];
    }
}

parameters{

    real<lower = 0, upper = (10 * pi())> w_rot;
    real<lower = 0, upper = 2 * pi()> initial_substellar_longitude;
    real<lower = 0, upper = 1> slices_of_albedo[number_of_slices];
}

transformed parameters{

    real substellar_longitude[number_of_data_points];
    real generated_lightcurve[number_of_data_points];
    
    for (t in 1:number_of_data_points){
        substellar_longitude[t] = get_substellar_longitude(times[t], initial_substellar_longitude, w_rot);
    }
    
    for (t in 1:number_of_data_points){
    
        generated_lightcurve[t] = 0;
        
        for (i in 1:number_of_slices){
            generated_lightcurve[t] += (get_albedo_sum_term(slices_of_albedo[i], substellar_longitude[t], phi_i[i], phi_i1[i]) / pi());
        }
    }
    
    //print("Generated Lightcurve: ", generated_lightcurve);
}

model{
    target += forwardModel_lpdf(lightcurve_data | generated_lightcurve, errors);
}
'''

sm = pystan.StanModel(model_code = ocode)
data_dict = dict(number_of_data_points = num_times, number_of_slices = m, lightcurve_data = albedo_lightcurve_data_with_errors, times = times, errors = errors_on_data, longitudinal_boundary_angles = longitudinal_boundary_angles)

the_best_op_log_prob = -100000000000.

for i in range(20000):

    completed_optimization = False

    while not completed_optimization:

        try:
            op = sm.optimizing(data = data_dict, init = init_values,\
                               tol_param = 1e-12, tol_rel_grad = 1e3, tol_grad = 1e-12, tol_obj = 1e-16, tol_rel_obj = 1e0)

            completed_optimization = True
            
            temp_log_p = 0
            
            generated_lightcurve = op['generated_lightcurve']
            
            for t in range(num_times):
                temp_log_p -= 0.5 * (((albedo_lightcurve_data_with_errors[t] - generated_lightcurve[t])**2) / (errors_on_data[t]**2))
            
            if temp_log_p > the_best_op_log_prob:
                
                the_best_op_log_prob = temp_log_p
                best_init = init_values
                best_op = op
                
        except RuntimeError:
            print("Optimization Failed, trying new random initial values.")
            
        finally:
            init_values = {'w_rot': (2. * np.pi) * (random.random() + 0.5),\
                           'initial_substellar_longitude': 2. * np.pi * random.random(),\
                           'slices_of_albedo': [random.random() for i in range(m)]}
            
            
#init_alpha : float, optional
    #For BFGS and LBFGS, default is 0.001.
#tol_obj : float, optional
    #For BFGS and LBFGS, default is 1e-12.
#tol_rel_obj : int, optional
    #For BFGS and LBFGS, default is 1e4.
#tol_grad : float, optional
    #For BFGS and LBFGS, default is 1e-8.
#tol_rel_grad : float, optional
    #For BFGS and LBFGS, default is 1e7.
#tol_param : float, optional
    #For BFGS and LBFGS, default is 1e-8
    
print(best_op)

# Plot initial optimized lightcurve

plt.figure()
plt.plot(times, best_op['generated_lightcurve'], label = "Optimized Lightcurve From Stan", ls = ':', alpha = 0.6)
plt.plot(times, albedo_lightcurve_data_no_errors, label = "Synthetic Data No Errors", ls = ':', alpha = 0.6)
plt.errorbar(times, albedo_lightcurve_data_with_errors, yerr = errors_on_data, label = "Synthetic Data With 2% Errors", ls = "--", alpha = 0.6)
plt.xlabel("Time (days)", fontsize = 15)
plt.ylabel("Albedo (1)", fontsize = 15)
plt.legend(fontsize = 11)
plt.savefig(fname = "stan_optimized_lightcurve.pdf")

# Begin sampling using initial values found in optimizer

input_pars = ['w_rot', 'initial_substellar_longitude', 'slices_of_albedo']
n_chains = cores_to_use

biwr = best_init['w_rot']
biisl = best_init['initial_substellar_longitude']
bisoa = best_init['slices_of_albedo']

input_init = [{'w_rot': biwr * (1. + random.gauss(0., biwr * 0.01)),\
               'initial_substellar_longitude': biisl * (1. + random.gauss(0., biisl * 0.01)),\
               'slices_of_albedo': [soa * (1. + random.gauss(0., soa * 0.01)) for soa in bisoa]} for i in range(n_chains)]

sf4m = sm.sampling(data = data_dict, pars = input_pars, chains = n_chains, iter = 4000, init = input_init, n_jobs = n_chains, control = {'max_treedepth': 20})

plt.figure()
plt.xlabel("", fontsize = 10)
plt.ylabel("", fontsize = 10)
arviz.plot_trace(sf4m, plot_kwargs = {"textsize": 10})
plt.savefig(fname = "stan_sampled_posteriors.pdf")

print(sf4m.stansummary(pars = input_pars, digits_summary = 3))