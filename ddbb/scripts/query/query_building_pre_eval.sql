-- Author: Alberto M. Esmoris Pena
-- Brief: SQL query to compute the pre-evaluation of the building
--          experiments. The pre-evaluation considers only a few point
--          clouds instead of the full dataset.

SELECT
        models.notes, models.id,
        sum(datasets.num_points) as "Num points",
        sum(datasets.num_points*global_resultsets.oa)/sum(datasets.num_points) as "OA",
        sum(datasets.num_points*global_resultsets.f1)/sum(datasets.num_points) as "F1",
        sum(datasets.num_points*global_resultsets.mcc)/sum(datasets.num_points) as "MCC"
FROM resultsets
        JOIN datasets on (resultsets.dataset_id = datasets.id)
        JOIN models on (resultsets.model_id = models.id)
        JOIN global_resultsets on (resultsets.global_resultset_id = global_resultsets.id)
WHERE datasets.name like any(array['%MERGE_260%', '%MERGE_68%', '%MERGE_216%', '%MERGE_21%'])
	and datasets.name like '%BUILDING'
GROUP BY models.id
ORDER BY "F1" desc
