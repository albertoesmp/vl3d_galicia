-- Author: Alberto M. Esmoris Pena
-- Brief: SQL query to compute the global evaluation of the building and
--          vegetation experiments. The global evaluation considers all the 
--          predicted point clouds for the building and vegetation 
--          classification task.
-- Brief: SQL query to compute the class-wise recount of the points in the
--          original class system (i.e., before transforming the set of classes
--          to represent a particular experiment).

-- Global class recount and percentage
SELECT classes.name as "Class name",
	sum(class_distributions.recount) as "Class recount",
	100*sum(class_distributions.recount)/sum(datasets.num_points) as "Percentage"
FROM datasets
	JOIN class_distributions on (datasets.id = class_distributions.dataset_id)
	JOIN classes on (class_distributions.class_id = classes.id)
WHERE datasets.name like '%_ORIGINAL' 
	--AND classes.name in ('Water', 'Low vegetation', 'Mid vegetation', 'High vegetation', 'Building', 'Ground')
GROUP BY classes.id
ORDER BY "Class recount"
