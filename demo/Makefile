clean:
	rm -rf test data npy resmlp.relay ./*.json resmlp-byoc.relay resmlp-compiled.relay

model:
	./1_get_relay_model.sh 4

eqsat:
	./2_run_eqsat.sh

compile_mod:
	./3_compile_model.sh 4 1

byoc:
	./run_byoc.sh resmlp.relay

run_mod:
	./4_run_model.sh 4 0

full-run: model eqsat run_mod
