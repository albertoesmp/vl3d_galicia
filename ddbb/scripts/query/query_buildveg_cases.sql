-- Author: Alberto M. Esmoris Pena
-- Brief: SQL query to compute the case by case evaluation of the building and
--          vegetation experiments. The global evaluation considers all the
--          predicted point clouds for the building and vegetation
--          classification task.

SELECT
    models.notes, models.id,
    datasets.name as "Point cloud",
    datasets.num_points as "Num points",
    datasets.num_points*global_resultsets.oa/datasets.num_points as "OA",
    datasets.num_points*global_resultsets.f1/datasets.num_points as "F1",
    datasets.num_points*global_resultsets.iou/datasets.num_points as "IoU",
    datasets.num_points*global_resultsets.mcc/datasets.num_points as "MCC"
FROM resultsets
    JOIN datasets on (resultsets.dataset_id = datasets.id)
    JOIN models on (resultsets.model_id = models.id)
    JOIN global_resultsets on (resultsets.global_resultset_id = global_resultsets.id)
WHERE models.id = 135
ORDER BY "F1" desc
