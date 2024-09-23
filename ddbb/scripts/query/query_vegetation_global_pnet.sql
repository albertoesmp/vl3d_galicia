-- Author: Alberto M. Esmoris Pena
-- Brief: SQL query to compute the global evaluation of the vegetation
--          experiments. The global evaluation considers all the predicted
--          point clouds for the vegetation classification task.


-- GALICIA
SELECT
        models.notes, models.id, 
        sum(datasets.num_points) as "Num points",
        sum(datasets.num_points*global_resultsets.oa)/sum(datasets.num_points) as "OA",
        sum(datasets.num_points*global_resultsets.p)/sum(datasets.num_points) as "P",
        sum(datasets.num_points*global_resultsets.r)/sum(datasets.num_points) as "R",
        sum(datasets.num_points*global_resultsets.f1)/sum(datasets.num_points) as "F1",
        sum(datasets.num_points*global_resultsets.iou)/sum(datasets.num_points) as "IoU",
        sum(datasets.num_points*global_resultsets.wp)/sum(datasets.num_points) as "wP",
        sum(datasets.num_points*global_resultsets.wr)/sum(datasets.num_points) as "wR",
        sum(datasets.num_points*global_resultsets.wf1)/sum(datasets.num_points) as "wF1",
        sum(datasets.num_points*global_resultsets.wiou)/sum(datasets.num_points) as "wIoU",
        sum(datasets.num_points*global_resultsets.mcc)/sum(datasets.num_points) as "MCC",
        sum(datasets.num_points*global_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
        JOIN datasets on (resultsets.dataset_id = datasets.id)
        JOIN models on (resultsets.model_id = models.id)
        JOIN global_resultsets on (resultsets.global_resultset_id = global_resultsets.id)
WHERE models.id = 211
GROUP BY models.id
ORDER BY "F1" desc




-- GALICIA WEST
SELECT
        models.notes, models.id, 
        sum(datasets.num_points) as "Num points",
        sum(datasets.num_points*global_resultsets.oa)/sum(datasets.num_points) as "OA",
        sum(datasets.num_points*global_resultsets.p)/sum(datasets.num_points) as "P",
        sum(datasets.num_points*global_resultsets.r)/sum(datasets.num_points) as "R",
        sum(datasets.num_points*global_resultsets.f1)/sum(datasets.num_points) as "F1",
        sum(datasets.num_points*global_resultsets.iou)/sum(datasets.num_points) as "IoU",
        sum(datasets.num_points*global_resultsets.wp)/sum(datasets.num_points) as "wP",
        sum(datasets.num_points*global_resultsets.wr)/sum(datasets.num_points) as "wR",
        sum(datasets.num_points*global_resultsets.wf1)/sum(datasets.num_points) as "wF1",
        sum(datasets.num_points*global_resultsets.wiou)/sum(datasets.num_points) as "wIoU",
        sum(datasets.num_points*global_resultsets.mcc)/sum(datasets.num_points) as "MCC",
        sum(datasets.num_points*global_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
        JOIN datasets on (resultsets.dataset_id = datasets.id)
        JOIN models on (resultsets.model_id = models.id)
        JOIN global_resultsets on (resultsets.global_resultset_id = global_resultsets.id)
        JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
        JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE models.id = 211 and 
    lower(geographic_regions.name) in ('a coruña', 'pontevedra')
GROUP BY models.id
ORDER BY "F1" desc




-- GALICIA EAST
SELECT
        models.notes, models.id, 
        sum(datasets.num_points) as "Num points",
        sum(datasets.num_points*global_resultsets.oa)/sum(datasets.num_points) as "OA",
        sum(datasets.num_points*global_resultsets.p)/sum(datasets.num_points) as "P",
        sum(datasets.num_points*global_resultsets.r)/sum(datasets.num_points) as "R",
        sum(datasets.num_points*global_resultsets.f1)/sum(datasets.num_points) as "F1",
        sum(datasets.num_points*global_resultsets.iou)/sum(datasets.num_points) as "IoU",
        sum(datasets.num_points*global_resultsets.wp)/sum(datasets.num_points) as "wP",
        sum(datasets.num_points*global_resultsets.wr)/sum(datasets.num_points) as "wR",
        sum(datasets.num_points*global_resultsets.wf1)/sum(datasets.num_points) as "wF1",
        sum(datasets.num_points*global_resultsets.wiou)/sum(datasets.num_points) as "wIoU",
        sum(datasets.num_points*global_resultsets.mcc)/sum(datasets.num_points) as "MCC",
        sum(datasets.num_points*global_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
        JOIN datasets on (resultsets.dataset_id = datasets.id)
        JOIN models on (resultsets.model_id = models.id)
        JOIN global_resultsets on (resultsets.global_resultset_id = global_resultsets.id)
        JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
        JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE models.id = 211 and 
    lower(geographic_regions.name) in ('lugo', 'ourense')
GROUP BY models.id
ORDER BY "F1" desc




-- A CORUÑA
SELECT
        models.notes, models.id, 
        sum(datasets.num_points) as "Num points",
        sum(datasets.num_points*global_resultsets.oa)/sum(datasets.num_points) as "OA",
        sum(datasets.num_points*global_resultsets.p)/sum(datasets.num_points) as "P",
        sum(datasets.num_points*global_resultsets.r)/sum(datasets.num_points) as "R",
        sum(datasets.num_points*global_resultsets.f1)/sum(datasets.num_points) as "F1",
        sum(datasets.num_points*global_resultsets.iou)/sum(datasets.num_points) as "IoU",
        sum(datasets.num_points*global_resultsets.wp)/sum(datasets.num_points) as "wP",
        sum(datasets.num_points*global_resultsets.wr)/sum(datasets.num_points) as "wR",
        sum(datasets.num_points*global_resultsets.wf1)/sum(datasets.num_points) as "wF1",
        sum(datasets.num_points*global_resultsets.wiou)/sum(datasets.num_points) as "wIoU",
        sum(datasets.num_points*global_resultsets.mcc)/sum(datasets.num_points) as "MCC",
        sum(datasets.num_points*global_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
        JOIN datasets on (resultsets.dataset_id = datasets.id)
        JOIN models on (resultsets.model_id = models.id)
        JOIN global_resultsets on (resultsets.global_resultset_id = global_resultsets.id)
        JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
        JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE models.id = 211 and 
    lower(geographic_regions.name) = 'a coruña'
GROUP BY models.id
ORDER BY "F1" desc




-- PONTEVEDRA
SELECT
        models.notes, models.id, 
        sum(datasets.num_points) as "Num points",
        sum(datasets.num_points*global_resultsets.oa)/sum(datasets.num_points) as "OA",
        sum(datasets.num_points*global_resultsets.p)/sum(datasets.num_points) as "P",
        sum(datasets.num_points*global_resultsets.r)/sum(datasets.num_points) as "R",
        sum(datasets.num_points*global_resultsets.f1)/sum(datasets.num_points) as "F1",
        sum(datasets.num_points*global_resultsets.iou)/sum(datasets.num_points) as "IoU",
        sum(datasets.num_points*global_resultsets.wp)/sum(datasets.num_points) as "wP",
        sum(datasets.num_points*global_resultsets.wr)/sum(datasets.num_points) as "wR",
        sum(datasets.num_points*global_resultsets.wf1)/sum(datasets.num_points) as "wF1",
        sum(datasets.num_points*global_resultsets.wiou)/sum(datasets.num_points) as "wIoU",
        sum(datasets.num_points*global_resultsets.mcc)/sum(datasets.num_points) as "MCC",
        sum(datasets.num_points*global_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
        JOIN datasets on (resultsets.dataset_id = datasets.id)
        JOIN models on (resultsets.model_id = models.id)
        JOIN global_resultsets on (resultsets.global_resultset_id = global_resultsets.id)
        JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
        JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE models.id = 211 and 
    lower(geographic_regions.name) = 'pontevedra'
GROUP BY models.id
ORDER BY "F1" desc




-- LUGO
SELECT
        models.notes, models.id, 
        sum(datasets.num_points) as "Num points",
        sum(datasets.num_points*global_resultsets.oa)/sum(datasets.num_points) as "OA",
        sum(datasets.num_points*global_resultsets.p)/sum(datasets.num_points) as "P",
        sum(datasets.num_points*global_resultsets.r)/sum(datasets.num_points) as "R",
        sum(datasets.num_points*global_resultsets.f1)/sum(datasets.num_points) as "F1",
        sum(datasets.num_points*global_resultsets.iou)/sum(datasets.num_points) as "IoU",
        sum(datasets.num_points*global_resultsets.wp)/sum(datasets.num_points) as "wP",
        sum(datasets.num_points*global_resultsets.wr)/sum(datasets.num_points) as "wR",
        sum(datasets.num_points*global_resultsets.wf1)/sum(datasets.num_points) as "wF1",
        sum(datasets.num_points*global_resultsets.wiou)/sum(datasets.num_points) as "wIoU",
        sum(datasets.num_points*global_resultsets.mcc)/sum(datasets.num_points) as "MCC",
        sum(datasets.num_points*global_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
        JOIN datasets on (resultsets.dataset_id = datasets.id)
        JOIN models on (resultsets.model_id = models.id)
        JOIN global_resultsets on (resultsets.global_resultset_id = global_resultsets.id)
        JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
        JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE models.id = 211 and 
    lower(geographic_regions.name) = 'lugo'
GROUP BY models.id
ORDER BY "F1" desc




-- OURENSE
SELECT
        models.notes, models.id, 
        sum(datasets.num_points) as "Num points",
        sum(datasets.num_points*global_resultsets.oa)/sum(datasets.num_points) as "OA",
        sum(datasets.num_points*global_resultsets.p)/sum(datasets.num_points) as "P",
        sum(datasets.num_points*global_resultsets.r)/sum(datasets.num_points) as "R",
        sum(datasets.num_points*global_resultsets.f1)/sum(datasets.num_points) as "F1",
        sum(datasets.num_points*global_resultsets.iou)/sum(datasets.num_points) as "IoU",
        sum(datasets.num_points*global_resultsets.wp)/sum(datasets.num_points) as "wP",
        sum(datasets.num_points*global_resultsets.wr)/sum(datasets.num_points) as "wR",
        sum(datasets.num_points*global_resultsets.wf1)/sum(datasets.num_points) as "wF1",
        sum(datasets.num_points*global_resultsets.wiou)/sum(datasets.num_points) as "wIoU",
        sum(datasets.num_points*global_resultsets.mcc)/sum(datasets.num_points) as "MCC",
        sum(datasets.num_points*global_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
        JOIN datasets on (resultsets.dataset_id = datasets.id)
        JOIN models on (resultsets.model_id = models.id)
        JOIN global_resultsets on (resultsets.global_resultset_id = global_resultsets.id)
        JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
        JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE models.id = 211 and 
    lower(geographic_regions.name) = 'ourense'
GROUP BY models.id
ORDER BY "F1" desc
