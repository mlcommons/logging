# benchmark dictionary
_ALL_RESULT_FILE_COUNTS = {
    'training': {
        'bert': 10,
        'dlrm': 5,
        'gnmt': 10,
        'maskrcnn': 5,
        'minigo': 10,
        'resnet': 5,
        'ssd': 5,
        'transformer': 10,
        'ncf': 10,
        'rnnt': 10,
        'unet3d': 40,
    },
    
    'hpc' : {
    'deepcam': 5,
        'cosmoflow': 10,
        'oc20': 10
    }
}


_ALL_ALLOWED_BENCHMARKS = {
    'training': {
        '0.6': [
            'resnet',
            'ssd',
            'maskrcnn',
            'gnmt',
            'transformer',
            'ncf',
            'minigo',
        ],
        
    '0.7': [
        'bert',
        'dlrm',
        'gnmt',
        'maskrcnn',
        'minigo',
        'resnet',
        'ssd',
        'transformer'
    ],
        
    '1.0': [
        'bert',
        'dlrm',
        'maskrcnn',
        'minigo',
        'resnet',
        'ssd',
        'rnnt',
        'unet3d',
    ],
    },
    
    'hpc': {
        '0.7': [
            'cosmoflow',
            'deepcam',
        ],
        
        '1.0': [
            'cosmoflow',
            'deepcam',
            'oc20',
        ],
    }
}
