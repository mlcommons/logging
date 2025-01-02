# benchmark dictionary
_ALL_RESULT_FILE_COUNTS = {
    'training': {
        'bert': 10,
        'dlrm': 5,
        'dlrm_dcnv2': 10,
        'gnmt': 10,
        'gpt3': 3,
        'maskrcnn': 5,
        'minigo': 10,
        'resnet': 5,
        'ssd': 5,
        'retinanet': 5,
        'stable_diffusion': 10,
        'transformer': 10,
        'ncf': 10,
        'rnnt': 10,
        'unet3d': 40,
        'gnn' : 10,
        'rgat': 10,  
        'llama2_70b_lora': 10,
    },
    
    'hpc' : {
        'deepcam': 5,
        'cosmoflow': 10,
        'oc20': 5,
        'openfold': 10,
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
    '1.1': [
        'bert',
        'dlrm',
        'maskrcnn',
        'minigo',
        'resnet',
        'ssd',
        'rnnt',
        'unet3d',
    ],
    '2.0': [
        'bert',
        'dlrm',
        'maskrcnn',
        'minigo',
        'resnet',
        'ssd',
        'rnnt',
        'unet3d',
    ],
    '2.1': [
        'bert',
        'dlrm',
        'maskrcnn',
        'minigo',
        'resnet',
        'ssd',
        'rnnt',
        'unet3d',
    ],
    '3.0': [
        'bert',
        'dlrm_dcnv2',
        'gpt3',
        'maskrcnn',
        'resnet',
        'ssd',
        'rnnt',
        'unet3d',
    ],
    '3.1': [
        'bert',
        'dlrm_dcnv2',
        'gpt3',
        'maskrcnn',
        'resnet',
        'ssd',
        'rnnt',
        'unet3d',
        'stable_diffusion'
    ],
    '4.0': [
        'bert',
        'dlrm_dcnv2',
        'gpt3',
        'resnet',
        'ssd',
        'unet3d',
        'stable_diffusion',
        'llama2_70b_lora',
        'stable_diffusion',
        'gnn'
    ],
    '4.1': [
        'bert',
        'dlrm_dcnv2',
        'gpt3',        
        'ssd',        
        'stable_diffusion',
        'llama2_70b_lora',
        'gnn'
    ],
    '5.0': [
        'bert',
        'dlrm_dcnv2',   
        'retinanet',        
        'stable_diffusion',
        'llama2_70b_lora',
        'rgat'
    ]    
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
        '2.0': [
            'cosmoflow',
            'deepcam',
            'oc20',
        ],
        '3.0': [
            'cosmoflow',
            'deepcam',
            'oc20',
            'openfold',
        ],
    }
}


def get_allowed_benchmarks(usage, ruleset):
    # check usage
    if usage not in _ALL_ALLOWED_BENCHMARKS:
        raise ValueError('usage {} not supported!'.format(usage))

    # check ruleset
    if ruleset not in _ALL_ALLOWED_BENCHMARKS[usage]:
        # try short version:
        ruleset_short = ".".join(ruleset.split(".")[:-1])
        if ruleset_short not in _ALL_ALLOWED_BENCHMARKS[usage]:
            raise ValueError('ruleset {} is not supported in {}'.format(ruleset, usage))
        allowed_benchmarks = _ALL_ALLOWED_BENCHMARKS[usage][ruleset_short]
    else:
        allowed_benchmarks = _ALL_ALLOWED_BENCHMARKS[usage][ruleset]

    return allowed_benchmarks


def get_result_file_counts(usage):
    if usage not in _ALL_RESULT_FILE_COUNTS:
        raise ValueError('usage {} not supported!'.format(usage))
    return _ALL_RESULT_FILE_COUNTS[usage]
