import os

os.environ["CC"] = "gcc"
os.environ["CXX"] = "g++"
os.environ["CXXFLAGS"] = "-O3 -mtune=native -march=native -Wno-unused-variable -Wno-unused-function"

import pystan
from matplotlib import pyplot as plt

import generate_data

m = 4
num_times = 20

synthetic_albedo_lightcurve_data, errors_on_data, times, longitudinal_boundary_angles = generate_data.generate_synthetic_albedo_lightcurve_data(number_Of_Slices = m, number_Of_Times = num_times, time_in_days_per_time_interval = 0.47)

ocode = """

functions{

    real get_substellar_longitude(real t, real initial_substellar_longitude, real w_rot, real w_orb){

        real substellar_longitude = initial_substellar_longitude - ((w_rot - w_orb) * t);

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
        
        if (albedo_sum_term <= 0){
            albedo_sum_term = 0;
        }

        return albedo_sum_term;
    }
    
    real forwardModel_lpdf(real[] lightcurve_data, int t, real[,] generated_lightcurve, real[] errors){
    
        int number_of_slices = dims(generated_lightcurve)[2];
        real returnedValue = 0;
        
        for (i in 1:number_of_slices){
            returnedValue -= 0.5 * ((((lightcurve_data[t] - generated_lightcurve[t,i])^2) / (errors[t]^2)) + log(2 * pi() * errors[t]^2));
        }
        
        print("Returned Value: ", returnedValue);
        
        return returnedValue;
    }
}

data{
    int<lower = 1> number_of_data_points;
    int<lower = 1> number_of_slices;
    real<lower = 0, upper = 1> lightcurve_data[number_of_data_points];
    real<lower = 0> times[number_of_data_points];
    real<lower = 0> errors[number_of_data_points];
    real<lower = 0, upper = 2 * pi()> longitudinal_boundary_angles[number_of_slices];
}

parameters{
    real<lower = 0> w_rot;
    real<lower = 0> w_orb;
    real<lower = 0, upper = 2 * pi()> initial_substellar_longitude;
    real<lower = 0, upper = 1> slices_of_albedo[number_of_slices];
}

transformed parameters{

    real substellar_longitude[number_of_data_points];
    real phi_i[number_of_slices];
    real phi_i1[number_of_slices];
    real generated_lightcurve[number_of_data_points, number_of_slices];
    
    for (t in 1:number_of_data_points){
      substellar_longitude[t] = get_substellar_longitude(times[t], initial_substellar_longitude, w_rot, w_orb);
    }
    
    for (i in 1:number_of_slices){
        
        phi_i[i] = longitudinal_boundary_angles[i];
        phi_i1[i] = longitudinal_boundary_angles[(i % number_of_slices) + 1];
        
    }
    
    for (t in 1:number_of_data_points){
        for (i in 1:number_of_slices){
            generated_lightcurve[t,i] = get_albedo_sum_term(slices_of_albedo[i], substellar_longitude[t], phi_i[i], phi_i1[i]) / pi();
        }
    }
}

model{
    for (t in 1:number_of_data_points){
        target += forwardModel_lpdf(lightcurve_data | t, generated_lightcurve, errors);
    }
}
"""

sm = pystan.StanModel(model_code=ocode)

op = sm.optimizing(data = dict(number_of_data_points = num_times, number_of_slices = m, lightcurve_data = synthetic_albedo_lightcurve_data, times = times, errors = errors_on_data, longitudinal_boundary_angles = longitudinal_boundary_angles))

plt.figure()
plt.plot(times, op, label = "Optimized Lightcurve From Stan")
plt.errorbar(times, synthetic_albedo_lightcurve_data, yerr = errors, label = "Synthetic Data With 2% Errors")
plt.xlabel("Time (days)", fontsize = 15)
plt.ylabel("Albedo (1)", fontsize = 15)
plt.legend(fontsize = 12)
plt.savefig(fname = "stan_hmc_lightcurve.pdf")