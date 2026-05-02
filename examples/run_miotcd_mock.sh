python tools/make_mock_traffic_data.py --dataset miotcd --output-dir demo_data/miotcd_mock
python -m tafdg.cli   --dataset miotcd   --data-root demo_data/miotcd_mock   --holdout-domain jpeg   --rounds 2   --num-clients 8   --clients-per-round 0.5   --local-epochs 1   --batch-size 8   --model tinycnn   --image-size 32   --output-dir outputs/miotcd_mock
