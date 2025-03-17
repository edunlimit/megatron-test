import subprocess

print('Converting checkpoints...')
command = [
    'bash', '/opt/ml/code/toolkits/model_checkpoints_convertor/qwen/hf2mcore_qwen2.5_convertor.sh',
    '7B',
    '/opt/ml/input/data/dataset/dataset/qwen2.5-7B',
    '/opt/ml/code/Qwen2.5-7B-hf-to-mcore-te-tp1-pp1',
    '1',
    '1',
    'fp16',
    'true',
    'false'
]

# Execute the command
convert_result = subprocess.run(command, check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True)
print('converting done')
if convert_result.stdout:
    logging.info("Shell Script STDOUT:\n" + convert_result.stdout)
if convert_result.stderr:
    logging.warning("Shell Script STDERR:\n" + convert_result.stderr)

print('pretraining...')
command = [
    'sh', '/opt/ml/code/examples/qwen2_5/run_mcore_qwen.sh',
    'dsw',
    '7B',
    '4',
    '8',
    '1e-5',
    '1e-6',
    '128',
    '128',
    'fp16',
    '2',
    '1',
    '1',
    'true',
    'true',
    'true',
    'false',
    'false',
    'false',
    '100000',
    '/opt/ml/input/data/dataset/dataset/',
    '/opt/ml/input/data/dataset/dataset/',
    '/opt/ml/code/Qwen2.5-7B-hf-to-mcore-te-tp1-pp1',
    '10000',
    '100',
    '/opt/ml/model/'
]

# Execute the command
result = subprocess.run(command, check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True)
if result.stdout:
    logging.info("Shell Script STDOUT:\n" + result.stdout)
if result.stderr:
    logging.warning("Shell Script STDERR:\n" + result.stderr)