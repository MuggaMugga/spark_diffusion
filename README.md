This is a module to build a diffusion map on a spark cluster. The basic set of function to create a diffusion map is as follows, assuming data is an nxm numpy matrix.

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
rows = sc.parallelize(data).map(lambda x: [np.float64(i).item() for i in x]).collect()
y = distributed_diffusion_map.create_map(steps=5,epsilon=2)
y.make_similarity_matrix_2(rows, sc)
sparse = y.make_d_matrix(sc)
y.normalize_matrix(sc)
y.calculate_eigenvalues_v2()
y.get_transformed_data()

After this the final diffusion map will be stored in y.dmap. Included in this project is a jupyter notebook that illustrates this process for a simple dataset.
If you want a simple environment to test this class I recommend the jupyter/pyspark-notebook. You will only need to install the sparkDiffusionMap module.
