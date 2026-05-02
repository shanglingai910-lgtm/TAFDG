python tools/make_mock_traffic_data.py --dataset gtsrb --output-dir demo_data/gtsrb_mock
python -m tafdg.cli   --dataset gtsrb   --data-root demo_data/gtsrb_mock   --holdout-domain fog   --rounds 2   --num-clients 8   --clients-per-round 0.5   --local-epochs 1   --batch-size 8   --model tinycnn   --image-size 32   --output-dir outputs/gtsrb_mock
