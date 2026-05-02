python tools/make_mock_traffic_data.py --dataset tt100k --output-dir demo_data/tt100k_mock
python -m tafdg.cli   --dataset tt100k   --data-root demo_data/tt100k_mock   --holdout-domain rain   --rounds 2   --num-clients 8   --clients-per-round 0.5   --local-epochs 1   --batch-size 8   --model tinycnn   --image-size 32   --output-dir outputs/tt100k_mock
