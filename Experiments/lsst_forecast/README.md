The analysis steps are documented with jupyter notebooks and python scripts.
Here is an overview of how the analysis is run.
WARNING: the numbering convention skips (don't ask me why)

Note: the first phase (steps 0-4) is focused on generating emulated lens models, 
and only needs to be run one time, but does take a good chunk of effort. 

If using provided emulated mass models, skip to step08, where the
cosmological inference is configured and run. 

STEP00: Lens Catalog with lambda_int,beta_ani
- Start from the modified OM10 catalog from Venkatraman et al. 2025
- Assign a lambda_int and beta_ani to each lens from the ground truth population
- Compute ground truth stellar velocity dispersions 
- Save to: 
    - DataVectors/om10_venkatraman25_withkin.csv

STEP01: Image Models
- Generate a simulated HST and LSST image for each lens
- Use pre-trained paltas NPE networks to produce approximate mass models
    from simulated images
- Save to: 
    - HST image models: DataVectors/gold/image_models.h5
    - LSST image models: DataVectors/silver/image_models.h5


STEP02: Ground truths (yes, again...
    had to account for cleaning up numerical instabilities in 
    image simulation)
- Compute ground truth fermat potential differences, time-delays, etc.
- Save to:
    - DataVectors/gold/truth_metadata.csv
    - DataVectors/silver/truth_metadata.csv

STEP03: Lens model posteriors: 
- Start from image_models.h5, truth_metadata.csv
- step 1: compute fermat potential difference 500x, once for each sample from the 
    image-based mass model
- step 2: re-assign the mean of the joint posterior over fermat pot. and 
    mass model parameters based on the ground truth ("de-biasing" ML posteriors)
- step 3: re-scale covariance matrices to emulate different modeling fidelities
- step 4: assign samples of beta_ani to each sample of fermat pot. + mass model 
    params (from a prior that will be accounted for later)
- Save to:
    - DataVectors/gold/dbl_posteriors_DEBIASED.h5
    - DataVectors/gold/dbl_posteriors_JWST_DEBIASED.h5
    - DataVectors/gold/dbl_posteriors_TDCOSMO2025_DEBIASED.h5
    - DataVectors/gold/quad_posteriors_DEBIASED.h5
    - DataVectors/gold/quad_posteriors_JWST_DEBIASED.h5
    - DataVectors/gold/quad_posteriors_TDCOSMO2025_DEBIASED.h5
    - DataVectors/silver/dbl_posteriors_DEBIASED.h5
    - DataVectors/silver/quad_posteriors_DEBIASED.h5

STEP04: compute predicted kinematic model
- step 1: compute associated Jeans model quantity: J(mass model params, beta_ani)
    for each sample. See step04_ifu_kinematics.sh, step04_kinematics.sh. This is
    done on a computing cluster, using the lenstronomy galkin module.
- Modifies:
    - DataVectors/gold/dbl_posteriors_DEBIASED.h5
    - DataVectors/gold/dbl_posteriors_JWST_DEBIASED.h5
    - DataVectors/gold/dbl_posteriors_TDCOSMO2025_DEBIASED.h5
    - DataVectors/gold/quad_posteriors_DEBIASED.h5
    - DataVectors/gold/quad_posteriors_JWST_DEBIASED.h5
    - DataVectors/gold/quad_posteriors_TDCOSMO2025_DEBIASED.h5
    - DataVectors/silver/dbl_posteriors_DEBIASED.h5
    - DataVectors/silver/quad_posteriors_DEBIASED.h5

NOW... we skip to step08
We only need to run steps 0-4 one time. Then, we can run step 8 many times, 
to perform many experiments. 

#######################################
Cosmological Inference Experiments Here
#######################################
STEP08: run the experiment
- step 1: create a configuration file for the experiment
    - ex: InferenceRuns/exp0_2_config.py
- step 2: emulate data vectors for that experiment (that take into account 
    the lens selection, time-delay precision. etc. that are specificied in the 
    experiment configuration file)
    - run: step08_datavectors.py
    - Save to: InferenceRuns/exp0_2/static_datavectors_seed6.json
- step 3: run the hierarchical inference (using those data vectors)
    - run: step08_inference.py
    - Save to: InferenceRuns/exp0_2/w0wa_seed6_backend.h5
- Save to: 
    - Static data vectors: InferenceRuns/exp0_2/static_datavectors_seed6.json
    - MCMC chain: InferenceRuns/exp0_2/w0wa_seed6_backend.h5

STEP09: analyze the chains
- See: Figures.ipynb

STEP10: redshift configuration experiment
- Step 1: re-produce data vectors for this particular experiment to 
    controlled fpd precisions
    - run: step10_produce_debiased_samps.py
    - Save to: InferenceRuns/FOM_vs_z/dv_dict_lens720_fp_prec_0.02.json
- Step 2: emulate data vectors & run inference at different redshifts
    - run: step10_redshift_test.py
- Save to: 
    - InferenceRuns/FOM_vs_z/redshift_zlens=X_zsrc=Y_backend.h5