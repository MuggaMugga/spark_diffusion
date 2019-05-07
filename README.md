This is a module to build a diffusion map on a spark cluster. The basic set of function to create a diffusion map is as follows, assuming data is an nxm numpy matrix.

sc = SparkContext.getOrCreate()<br />
spark = SparkSession(sc)<br />
rows = sc.parallelize(data).map(lambda x: [np.float64(i).item() for i in x]).collect()<br />
y = distributed_diffusion_map.create_map(steps=5,epsilon=2)<br />
y.make_similarity_matrix_2(rows, sc)<br />
sparse = y.make_d_matrix(sc)<br />
y.normalize_matrix(sc)<br />
y.calculate_eigenvalues_v2()<br />
y.get_transformed_data()<br />

After this the final diffusion map will be stored in y.dmap. Included in this project is a jupyter notebook that illustrates this process for a simple dataset.
If you want a simple environment to test this class I recommend the jupyter/pyspark-notebook docker container. You will only need to install the sparkDiffusionMap module.
