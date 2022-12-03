clean:
	rm -rf test/__pycache__ && \
	rm -rf .pytest_cache && \
	rm -rf __pycache__ && \
	rm -rf models/__pycache__ && \
	rm -rf *.egg-info && \
	rm -rf mlruns

unit: clean
	source .venv/bin/activate && \
	pytest test --log-cli-level=INFO

horovod: clean
	pytest test/test_horovod.py --log-cli-level=INFO

train: clean
	pytest test/test_train.py --log-cli-level=INFO