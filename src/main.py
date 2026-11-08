from argparse import ArgumentParser

if __name__ == '__main__':
    # 定义参数解析器
    parser = ArgumentParser(usage='usage: main.py action')
    # 指定action参数的候选项
    parser.add_argument('action', choices=['train', 'evaluate', 'predict', 'preprocess', 'run_app'])
    # 解析参数
    args = parser.parse_args()
    # 取出当前操作
    action = args.action

    match action:
        case 'preprocess':
            from process.preprocess import preprocess
            preprocess()
        case 'train':
            from runner.train import train
            train()
        case 'predict':
            from runner.predict import predict
            predict()
        case 'evaluate':
            from runner.evaluate import evaluate
            evaluate()
        case 'run_app':
            from web.app import run_app
            run_app()