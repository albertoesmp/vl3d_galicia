-- Author: Alberto M. Esmoris Pena
-- Brief: Script to make some baseline inserts into the catadb database
-- Database: catadb

-- TABLE: modelers
INSERT INTO modelers (name, description)
    VALUES ('VL3D Galicia Team', 'Team for processing Galicia ALS data from PNOA-II dataset')
    ON CONFLICT DO NOTHING;

-- TABLE: projects
INSERT INTO projects (name, description)
    VALUES (
        'VL3D Galicia on PNOA-II ALS data',
        'Vegetation and structures classification on the whole region of Galicia.'
        'The point clouds are taken from the PNOA-II dataset.'
    ) ON CONFLICT DO NOTHING;

-- TABLE: project_modelers
INSERT INTO project_modelers (project_id, modeler_id)
    VALUES (
        (SELECT id FROM projects WHERE name='VL3D Galicia on PNOA-II ALS data'),
        (SELECT id FROM modelers WHERE name='VL3D Galicia Team')
    )
    ON CONFLICT DO NOTHING;

-- TABLE: data_domains
INSERT INTO data_domains (name, description)
    VALUES (
        'ALS',
        'Aerial Laser Scanning data. Typically acquired with flying platforms like helicopters equipped with LiDAR.'
    ),(
        'TLS',
        'Terrestrial Laser Scanning data. Typically acquired with a tripod where a LiDAR sensor is mounted.'
    ),(
        'ULS',
        'Unmanned Laser Scanning data. Typically acquired with a drone equipped with LiDAR.'
    ),(
        'MLS',
        'Mobile Laser Scanning data. Typically acquired with a car or a van equipped with LiDAR.'
    ),(
        'HMLS',
        'Hand-packed Mobile Laser Scanning data. The LiDAR sensor is typically carried by human or a robot.'
    ) ON CONFLICT DO NOTHING;

INSERT INTO target_domains(name, description)
    VALUES(
        'Vegetation',
        'Any vegetation.'
    ),(
        'Low vegetation',
        'Some times it is ground-level vegetation but it can also refer to vegetation below a given height threshold,'
        'e.g., vegetation below 1.5 meters.'
    ),(
        'Mid vegetation',
        'A vegetation label common in some classification tasks that differentiate between low, mid, and high vegetation.'
    ),(
        'High vegetation',
        'Some times it means trees but it can also refer to vegetation above a given height threshold,'
        'e.g., vegetation above 7 meters.'
    ),(
        'Structure',
        'Any type of artificial structure.'
    ),(
        'Building',
        'Buildings are a particular type of structure.'
    ) ON CONFLICT DO NOTHING;

INSERT INTO domains(data_domain_id, target_domain_id)
    VALUES(
        (SELECT id from data_domains WHERE name='ALS'),
        (SELECT id from target_domains WHERE name='Vegetation')
    ),(
        (SELECT id from data_domains WHERE name='ALS'),
        (SELECT id from target_domains WHERE name='Low vegetation')
    ),(
        (SELECT id from data_domains WHERE name='ALS'),
        (SELECT id from target_domains WHERE name='Mid vegetation')
    ),(
        (SELECT id from data_domains WHERE name='ALS'),
        (SELECT id from target_domains WHERE name='High vegetation')
    ),(
        (SELECT id from data_domains WHERE name='ALS'),
        (SELECT id from target_domains WHERE name='Structure')
    ),(
        (SELECT id from data_domains WHERE name='ALS'),
        (SELECT id from target_domains WHERE name='Building')
    ) ON CONFLICT DO NOTHING;

INSERT INTO project_domains(project_id, domain_id)
    VALUES(
        (SELECT id FROM projects WHERE name='VL3D Galicia on PNOA-II ALS data'),
        (SELECT id from domains WHERE 
            data_domain_id in (
                SELECT id from data_domains WHERE name='ALS'
            ) and target_domain_id in(
                SELECT id from target_domains WHERE name='Vegetation'
            )
        )
    ),(
        (SELECT id FROM projects WHERE name='VL3D Galicia on PNOA-II ALS data'),
        (SELECT id from domains WHERE 
            data_domain_id in (
                SELECT id from data_domains WHERE name='ALS'
            ) and target_domain_id in(
                SELECT id from target_domains WHERE name='Low vegetation'
            )
        )
    ),(
        (SELECT id FROM projects WHERE name='VL3D Galicia on PNOA-II ALS data'),
        (SELECT id from domains WHERE 
            data_domain_id in (
                SELECT id from data_domains WHERE name='ALS'
            ) and target_domain_id in(
                SELECT id from target_domains WHERE name='Mid vegetation'
            )
        )
    ),(
        (SELECT id FROM projects WHERE name='VL3D Galicia on PNOA-II ALS data'),
        (SELECT id from domains WHERE 
            data_domain_id in (
                SELECT id from data_domains WHERE name='ALS'
            ) and target_domain_id in(
                SELECT id from target_domains WHERE name='High vegetation'
            )
        )
    ),(
        (SELECT id FROM projects WHERE name='VL3D Galicia on PNOA-II ALS data'),
        (SELECT id from domains WHERE 
            data_domain_id in (
                SELECT id from data_domains WHERE name='ALS'
            ) and target_domain_id in(
                SELECT id from target_domains WHERE name='Structure'
            )
        )
    ),(
        (SELECT id FROM projects WHERE name='VL3D Galicia on PNOA-II ALS data'),
        (SELECT id from domains WHERE 
            data_domain_id in (
                SELECT id from data_domains WHERE name='ALS'
            ) and target_domain_id in(
                SELECT id from target_domains WHERE name='Building'
            )
        )
    ) ON CONFLICT DO NOTHING;

-- TABLE: framework_names
INSERT INTO framework_names (name, description)
    VALUES('VL3D', 'The VL3D framework for AI applied to point clouds.')
    ON CONFLICT DO NOTHING;

-- TABLE: frameworks
INSERT INTO frameworks (framework_name_id, version, notes)
    VALUES(
        (SELECT id FROM framework_names WHERE name='VL3D'),
        'Alpha prerelease 0.1',
        'The version of the VL3D framework with the first implementation of hierarchical autoencoders.'
    ) ON CONFLICT DO NOTHING;

-- TABLE: machines
INSERT INTO machines (
        name,
        cpu, cpu_max_freq, cpu_max_cores,
        gpu, gpu_max_freq, gpu_max_cores, gpu_max_mem,
        ram, ram_max_mem,
        notes
    ) VALUES(
        'FinisTerrae-III A100',
        'Intel Xeon Ice Lake 8352Y (x2)', 2200000000, 32,
        'NVIDIA A100-PCIE (x2)', 1410000000, 6912, 40000000000,
        'RAM (expected 256GB)', 247000000,
        'The FinisTerrae-III nodes with NVIDIA A100 GPUs'
    ) ON CONFLICT DO NOTHING;

-- TABLE: metadatasets
INSERT INTO metadatasets (name, description, url, owner, open_access, unrestricted_open_access)
    VALUES ('PNOA-II GALICIA',
        'Galicia region from the ALS PNOA-II dataset',
        'https://pnoa.ign.es/pnoa-lidar/segunda-cobertura',
        'Ministerio de Transported y Movilidad Sostenible, Gobierno de España',
        true,
        false
    ) ON CONFLICT DO NOTHING;

-- TABLE: classes
INSERT INTO classes (name, description)
    VALUES(
        'Vegetation',
        'Any vegetation.'
    ),(
        'Low vegetation',
        'Some times it is ground-level vegetation but it can also refer to vegetation below a given height threshold,'
        'e.g., vegetation below 1.5 meters.'
    ),(
        'Mid vegetation',
        'A vegetation label common in some classification tasks that differentiate between low, mid, and high vegetation.'
    ),(
        'High vegetation',
        'Some times it means trees but it can also refer to vegetation above a given height threshold,'
        'e.g., vegetation above 7 meters.'
    ),(
        'Structure',
        'Any type of artificial structure.'
    ),(
        'Building',
        'Buildings are a particular type of structure.'
    ),(
        'Noise',
        'Points that do not bring reliable information.'
    ),(
        'Overlap',
        'Points that belong to a region covered by many passes (typically the worst quality points are labeled as overlap).'
    ),(
        'Ground',
        'Points representing the ground or terrain.'
    ),(
        'Water',
        'Points acquired by scanning water regions like seas or rivers.'
    ),(
        'Bridge',
        'Points representing manmade bridges.'
    ),( 
        'Other',
        'Points that do not belong to any of the other classes in the dataset.'
    ),(
        'Ignore',
        'Points that can be safely ignored by a classifier (e.g., unlabeled points for semisupervised learning).'
    ),(
        'Unclassified',
        'Points for which there is no known class. Note that a point labeled as Ignore may be just a convenience '
        'including points for which labels are known. Unclassified implies that there is no known class for the point.'
    ) ON CONFLICT DO NOTHING;

-- TABLE: plots
INSERT INTO plots (name, description)
    VALUES(
        'Class reduction distribution',
        'Bar diagrams representing the distributions of both the original and the reduced classes.'
    ),(
        'Model graph',
        'A graph-based representation of the deep learning architecture.'
    ),(
        'Training confusion matrix',
        'The confusion matrix on the training data.'
    ),(
        'Training receptive fields distribution',
        'The representation of the receptive fields generated from the training dataset.'
    ),(
        'Class distribution',
        'Bar diagrams representing the distributions of the reference and predicted classes on the training dataset.'
    ),(
        'Training history summary',
        'Plot representing each measurement computed during training (e.g., the loss function).'
    ),(
        'Categorical accuracy history',
        'Plot representing the categorical accuracy training history.'
    ),(
        'Loss history',
        'Plot representing the training history of the loss function.'
    ),
    (
        'Learning rate history',
        'Plot representing the learning rate training history.'
    ),(
        'Validation confusion matrix',
        'The confusion matrix on the validation data.'
    ),(
        'Point-wise entropy',
        'The point-wise entropy violin plot.'
    ),(
        'Class ambiguity',
        'The point-wise class ambiguity violin plot.'
    ),(
        'Weighted entropy',
        'The point-wise weighted entropy violing plot.'
    ),(
        'Cluster-wise entropy',
        'The cluster-wise entropy violin plot.'
    ),(
        'Validation receptive fields distribution',
        'The representation of the receptive fields generated from the validation dataset.'
    ) ON CONFLICT DO NOTHING;

-- TABLE: plot_formats
INSERT INTO plot_formats(name, description)
    VALUES(
        'PNG',
        'Plot stored in Portable Newtork Graphics (PNG) format.'
    ),
    (
        'JPG',
        'Plot stored in Joing Photographic Experts Group (JPG/JPEG) format.'
    ),(
        'GIF',
        'Plot stored in Graphic Interchange Format (GIF).'
    ),(
        'BMP',
        'Plot stored in Bitmap (BMP) format.'
    ),(
        'SVG',
        'Plot stored in Scalable Vector Graphics (SVG) format.'
    ),(
        'GeoTIFF',
        'Plot stored in Geographic Tag Image File Format (GeoTIFF).'
    ) ON CONFLICT DO NOTHING;

-- TABLE: uncertainty_metrics
INSERT INTO uncertainty_metrics(name, description)
    VALUES(
        'Point-wise entropy',
        'The point-wise Shannon entropy.'
    ),(
        'Weighted point-wise entropy',
        'The weighted (by number of cases) point-wise Shannon entropy.'
    ),(
        'Cluster-wise entropy',
        'The Shannon entropy computed for clusters of points (typically clustered in the feature space).'
    ),
    (
        'Class ambiguity',
        'A measurement of the difference between the max and the second max likelihoods.'
        'The greater this difference, the lower the class ambiguity.'
    )
    ON CONFLICT DO NOTHING;

-- TABLE: geographic_regions
INSERT INTO geographic_regions(name, admin_level, geocode_area) VALUES
    ('Galicia', 4, 'Galicia'),
    ('A Coruña', 6, 'Coruña'),
    ('Pontevedra', 6, 'Pontevedra'),
    ('Lugo', 6, 'Lugo'),
    ('Ourense', 6, 'Ourense')
    ON CONFLICT DO NOTHING;


-- TABLE: model_families
INSERT INTO model_families(name, description) VALUES
    ('Sequential', 'A sequential neural network architecture.'),
    ('Hierarchical autoencoder', 'A hierarchical autoencoder neural network architecture.')
    ON CONFLICT DO NOTHING;

-- TABLE: model_subfamilies
INSERT INTO model_subfamilies(name, description) VALUES
    ('PointNet', 'A model based on the PointNet operator.'),
    ('KPConv', 'A model baised on Kernel-Point Convolutions.')
    ON CONFLICT DO NOTHING;

-- TABLE: tasks
INSERT INTO tasks (type, description) VALUES
    ('Point-wise classification', 'Classify each point in the point cloud.'),
    ('Point-wise regression', 'Predict a continuous value for each point in the point cloud.')
    ON CONFLICT DO NOTHING;

-- TABLE: model_types
INSERT INTO model_types(specification, family_id, subfamily_id, notes) VALUES
    (
        '{}',
        (SELECT id FROM model_families WHERE LOWER(name) like 'sequential'),
        (SELECT id FROM model_subfamilies WHERE LOWER(name) like 'pointnet'),
        'A null model type that should not be used unless for development, debugging, and bug fixing purposes.'
    ) ON CONFLICT DO NOTHING;

-- TABLE: models
INSERT INTO models(model_type_id, framework_id, notes) VALUES
    (1, 1, 'Null model for development, testing, and debugging purposes only.')
    ON CONFLICT DO NOTHING;