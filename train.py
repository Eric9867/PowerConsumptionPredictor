import numpy as np
import model3

if __name__ == '__main__':

    models = {}
    N_ITERATIONS = 5
    
    all_inputs = ['Month', 'Day', 'Time', 'Temperature', 'Humidity', 'Wind Speed', 'general diffuse flows', 'diffuse flows']
    for input in all_inputs:
        inputs = all_inputs[:]
        inputs.remove(input)
        for i in range(N_ITERATIONS):    
            if input not in models:
                models[input] = []
            mdl, a, b, c = model3.train_generic(model3.create_model((len(inputs),)), inputs)
            models[input].append([a, b, c])

    for key in models:
        models[key] = np.mean(np.asarray(models[key]), axis=0)

    for key, val in models.items():
        print(f'{key} \t{val}')



# R^2 Scores

# Month                   [0.69024732 0.59163403 0.56729844]
# Day                     [0.74189464 0.70418084 0.78438368]
# Time                    [0.27459664 0.33162596 0.59183463]
# Temperature             [0.72433601 0.68375991 0.72446511]
# Humidity                [0.70385884 0.67672056 0.75766386]
# Wind Speed              [0.72495343 0.67913559 0.75361359]
# general diffuse flows   [0.66590739 0.6423805  0.73402222]
# diffuse flows           [0.70308413 0.66979654 0.7556972 ]
