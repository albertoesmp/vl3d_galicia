-- Author: Alberto M. Esmoris Pena
-- Brief: SQL query to compute the class-wise evaluation of the low-mid-high
--          vegetation experiments. The class-wise evaluation considers all the 
--          predicted point clouds for the low-mid-high vegetation 
--          classification task.


-- GALICIA
SELECT
    sum(datasets.num_points) as "Num points",
    max(classes.name),
    sum(datasets.num_points*classwise_resultsets.p)/sum(datasets.num_points) as "P",
    sum(datasets.num_points*classwise_resultsets.r)/sum(datasets.num_points) as "R",
    sum(datasets.num_points*classwise_resultsets.f1)/sum(datasets.num_points) as "F1",
    sum(datasets.num_points*classwise_resultsets.iou)/sum(datasets.num_points) as "IoU",
    sum(datasets.num_points*classwise_resultsets.mcc)/sum(datasets.num_points) as "MCC",
    sum(datasets.num_points*classwise_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
    JOIN datasets on (resultsets.dataset_id = datasets.id)
    JOIN models on (resultsets.model_id = models.id)
    JOIN classwise_resultsets on (resultsets.id = classwise_resultsets.resultset_id)
    JOIN classes on (classwise_resultsets.class_id = classes.id)
WHERE models.id = 251
GROUP BY classes.id








-- GALICIA WEST
SELECT
    sum(datasets.num_points) as "Num points",
	max(classes.name),
	sum(datasets.num_points*classwise_resultsets.p)/sum(datasets.num_points) as "P",
	sum(datasets.num_points*classwise_resultsets.r)/sum(datasets.num_points) as "R",
	sum(datasets.num_points*classwise_resultsets.f1)/sum(datasets.num_points) as "F1",
	sum(datasets.num_points*classwise_resultsets.iou)/sum(datasets.num_points) as "IoU",
	sum(datasets.num_points*classwise_resultsets.mcc)/sum(datasets.num_points) as "MCC",
	sum(datasets.num_points*classwise_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
    JOIN datasets on (resultsets.dataset_id = datasets.id)
    JOIN models on (resultsets.model_id = models.id)
    JOIN classwise_resultsets on (resultsets.id = classwise_resultsets.resultset_id)
	JOIN classes on (classwise_resultsets.class_id = classes.id)
	JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
	JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE models.id = 251 and
	lower(geographic_regions.name) in ('a coruña', 'pontevedra')
GROUP BY classes.id
	




-- GALICIA EAST
SELECT
    sum(datasets.num_points) as "Num points",
    max(classes.name),
    sum(datasets.num_points*classwise_resultsets.p)/sum(datasets.num_points) as "P",
    sum(datasets.num_points*classwise_resultsets.r)/sum(datasets.num_points) as "R",
    sum(datasets.num_points*classwise_resultsets.f1)/sum(datasets.num_points) as "F1",
    sum(datasets.num_points*classwise_resultsets.iou)/sum(datasets.num_points) as "IoU",
    sum(datasets.num_points*classwise_resultsets.mcc)/sum(datasets.num_points) as "MCC",
    sum(datasets.num_points*classwise_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
    JOIN datasets on (resultsets.dataset_id = datasets.id)
    JOIN models on (resultsets.model_id = models.id)
    JOIN classwise_resultsets on (resultsets.id = classwise_resultsets.resultset_id)
    JOIN classes on (classwise_resultsets.class_id = classes.id)
    JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
    JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE models.id = 251 and
    lower(geographic_regions.name) in ('lugo', 'ourense')
GROUP BY classes.id




-- A CORUÑA
SELECT
    sum(datasets.num_points) as "Num points",
	max(classes.name),
	sum(datasets.num_points*classwise_resultsets.p)/sum(datasets.num_points) as "P",
	sum(datasets.num_points*classwise_resultsets.r)/sum(datasets.num_points) as "R",
	sum(datasets.num_points*classwise_resultsets.f1)/sum(datasets.num_points) as "F1",
	sum(datasets.num_points*classwise_resultsets.iou)/sum(datasets.num_points) as "IoU",
	sum(datasets.num_points*classwise_resultsets.mcc)/sum(datasets.num_points) as "MCC",
	sum(datasets.num_points*classwise_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
    JOIN datasets on (resultsets.dataset_id = datasets.id)
    JOIN models on (resultsets.model_id = models.id)
    JOIN classwise_resultsets on (resultsets.id = classwise_resultsets.resultset_id)
	JOIN classes on (classwise_resultsets.class_id = classes.id)
	JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
	JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE models.id = 251 and
	lower(geographic_regions.name) = 'a coruña'
GROUP BY classes.id
	




-- PONTEVEDRA
SELECT
    sum(datasets.num_points) as "Num points",
    max(classes.name),
    sum(datasets.num_points*classwise_resultsets.p)/sum(datasets.num_points) as "P",
    sum(datasets.num_points*classwise_resultsets.r)/sum(datasets.num_points) as "R",
    sum(datasets.num_points*classwise_resultsets.f1)/sum(datasets.num_points) as "F1",
    sum(datasets.num_points*classwise_resultsets.iou)/sum(datasets.num_points) as "IoU",
    sum(datasets.num_points*classwise_resultsets.mcc)/sum(datasets.num_points) as "MCC",
    sum(datasets.num_points*classwise_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
    JOIN datasets on (resultsets.dataset_id = datasets.id)
    JOIN models on (resultsets.model_id = models.id)
    JOIN classwise_resultsets on (resultsets.id = classwise_resultsets.resultset_id)
    JOIN classes on (classwise_resultsets.class_id = classes.id)
    JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
    JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE models.id = 251 and
    lower(geographic_regions.name) = 'pontevedra'
GROUP BY classes.id




-- LUGO
SELECT
    sum(datasets.num_points) as "Num points",
    max(classes.name),
    sum(datasets.num_points*classwise_resultsets.p)/sum(datasets.num_points) as "P",
    sum(datasets.num_points*classwise_resultsets.r)/sum(datasets.num_points) as "R",
    sum(datasets.num_points*classwise_resultsets.f1)/sum(datasets.num_points) as "F1",
    sum(datasets.num_points*classwise_resultsets.iou)/sum(datasets.num_points) as "IoU",
    sum(datasets.num_points*classwise_resultsets.mcc)/sum(datasets.num_points) as "MCC",
    sum(datasets.num_points*classwise_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
    JOIN datasets on (resultsets.dataset_id = datasets.id)
    JOIN models on (resultsets.model_id = models.id)
    JOIN classwise_resultsets on (resultsets.id = classwise_resultsets.resultset_id)
    JOIN classes on (classwise_resultsets.class_id = classes.id)
    JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
    JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE models.id = 251 and
    lower(geographic_regions.name) = 'lugo'
GROUP BY classes.id




-- OURENSE
SELECT
    sum(datasets.num_points) as "Num points",
    max(classes.name),
    sum(datasets.num_points*classwise_resultsets.p)/sum(datasets.num_points) as "P",
    sum(datasets.num_points*classwise_resultsets.r)/sum(datasets.num_points) as "R",
    sum(datasets.num_points*classwise_resultsets.f1)/sum(datasets.num_points) as "F1",
    sum(datasets.num_points*classwise_resultsets.iou)/sum(datasets.num_points) as "IoU",
    sum(datasets.num_points*classwise_resultsets.mcc)/sum(datasets.num_points) as "MCC",
    sum(datasets.num_points*classwise_resultsets.kappa)/sum(datasets.num_points) as "Kappa"
FROM resultsets
    JOIN datasets on (resultsets.dataset_id = datasets.id)
    JOIN models on (resultsets.model_id = models.id)
    JOIN classwise_resultsets on (resultsets.id = classwise_resultsets.resultset_id)
    JOIN classes on (classwise_resultsets.class_id = classes.id)
    JOIN dataset_regions on (datasets.id = dataset_regions.dataset_id)
    JOIN geographic_regions on (dataset_regions.region_id = geographic_regions.id)
WHERE models.id = 251 and
    lower(geographic_regions.name) = 'ourense'
GROUP BY classes.id
	
