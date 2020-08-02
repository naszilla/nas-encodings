
param_str=twentyfive

# set parameters based on the param string
if [ $param_str = test ]; then
	num_init=2
	start_iteration=2
	end_iteration=10
	epochs=1
	experiment_name=ls_darts_test
fi
if [ $param_str = twentyfive ]; then
	num_init=10
	start_iteration=10
	end_iteration=200
	epochs=25
	experiment_name=ls_darts_twentyfive
fi
if [ $param_str = fifty ]; then
	num_init=10
	start_iteration=10
	end_iteration=100
	epochs=25
	experiment_name=ls_darts_fifty
fi

for query in $(seq $start_iteration $end_iteration)
do 

	echo about to run local search round $query
	python ls_darts/local_search_runner.py --experiment_name $experiment_name \
		--query $query --num_init $num_init

	untrained_filepath=$experiment_name/untrained_spec\_$query.pkl
	trained_filepath=$experiment_name/trained_spec\_$query.pkl

	echo about to train architecture $query

	python train_arch_runner.py --untrained_filepath $untrained_filepath \
	--trained_filepath $trained_filepath --epochs $epochs >> training.out

	echo finished iteration $query
done

