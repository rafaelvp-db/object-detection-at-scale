clean:
	rm -rf test/__pycache__ && \
	rm -rf .pytest_cache && \
	rm -rf __pycache__ && \
	rm -rf models/__pycache__ && \
	rm -rf *.egg-info && \
	rm -rf mlruns