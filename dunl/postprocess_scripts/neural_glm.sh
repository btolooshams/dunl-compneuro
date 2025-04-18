


res_path_list="../results/local_raised_cosine_5_bases_res25ms ../results/local_raised_cosine_2_bases_res25ms ../results/local_raised_nonlin_cosine_5_bases_res25ms ../results/local_raised_nonlin_cosine_2_bases_res25ms"

for res_path in $res_path_list
do
python postprocess_scripts/plot_neuralglm_dopamine_spiking_eshel_uchida_code.py \
    --res-path=$res_path

python postprocess_scripts/plot_neuralglm_dopamine_spiking_eshel_uchida_base.py \
    --res-path=$res_path

python postprocess_scripts/plot_neuralglm_dopamine_spiking_eshel_uchida_rec.py \
    --res-path=$res_path

done



