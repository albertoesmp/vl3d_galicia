-- Author: Alberto M. Esmoris Pena
-- Brief: SQL query to compute the pre-evaluation of the low, mid, and high
--          vegetation experiments. The pre-evaluation considers only a few
--          point clouds instead of the full dataset.

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
WHERE datasets.name like any(array['%MERGE_234%', '%MERGE_97%', '%MERGE_239%', '%MERGE_189%'])
        and datasets.name like '%LMH_VEGETATION'
GROUP BY models.id
ORDER BY "F1" desc
