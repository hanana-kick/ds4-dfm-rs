import fs from 'node:fs';
function edit(p,f){fs.writeFileSync(p,f(fs.readFileSync(p,'utf8')));}
function once(s,a,b){if(s.split(a).length!==2)throw Error('anchor '+a);return s.replace(a,b);}
edit('crates/ds4-core/src/v41_run/batch/mtp.rs',s=>s+fs.readFileSync('tests/mtp-rust-tests.fragment','utf8'));
edit('crates/ds4-server/src/serve_v41.rs',s=>{
 s=once(s,'if (offered > 0) {','if offered > 0 {');
 return once(s,`            if state.done || row + 1 == rows
                || state.pending != state.native.verify_candidate(row + 1) { break; }`,
 `            if !state.native.verify_continue(row, state.pending, state.done) { break; }`);
});
edit('Makefile',s=>s+`
.PHONY: test-ds41-mtp-bridge
test-ds41-mtp-bridge:
	$(CC) $(CFLAGS) -O0 -ffunction-sections -fdata-sections -I. tests/test_ds41_mtp_bridge.c -Wl,--gc-sections -o /tmp/ds41-mtp-bridge -lm -pthread
	/tmp/ds41-mtp-bridge

gate-ds41: test-ds41-mtp-bridge
`);
edit('.github/workflows/ds41-port-checks.yml',s=>once(s,'test-ds41-batch-lifetime test-ds41-spec-plan','test-ds41-batch-lifetime test-ds41-spec-plan test-ds41-mtp-bridge'));
console.log('Added MTP transaction lifetime and target-sampler equivalence checks.');
