from parse import parse_args
args = parse_args()

hyper_params = {
	# COMMON
	'dataset': args.dataset, 
	'seed': args.seed,
	'model': args.model,

	# SVD-AE
	'k': args.k,
	'load': args.load,

	# Inifinite-AE
	'lamda': args.lamda, # Only used if grid_search_lamda == False
	'float64': False,
	'depth': 1,
	'grid_search_lamda': args.grid_search,
	'user_support': -1, #  Number of users to keep (randomly) & -1 implies use all users
}
