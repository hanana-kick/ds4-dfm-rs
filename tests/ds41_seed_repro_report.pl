#!/usr/bin/env perl
# Offline DS41 seed sweep report: no model, network, or Python required.
use strict;
use warnings;
use JSON::PP qw(decode_json encode_json);
use Encode qw(decode);

sub success {
    my ($v) = @_;
    return ref($v) eq 'HASH'
        && ($v->{status} // '') eq 'ok'
        && ($v->{finish_reason} // '') eq 'stop'
        && $v->{final_answer_present}
        && !$v->{reasoning_loop} && !$v->{answer_loop};
}

sub outcome {
    my ($a, $b, $equal) = @_;
    return 'missing' unless ref($a) eq 'HASH' && ref($b) eq 'HASH';
    return 'missing' if ($a->{status} // '') eq 'response_error'
        || ($b->{status} // '') eq 'response_error';
    return 'pass' if success($a) && success($b) && $equal;
    return 'nondeterministic' if success($a) && success($b);
    return $equal ? 'reproducible_failure' : 'varying_failure';
}

# Native V4.1 uses time/getpid/clock when the requested seed equals 0.
# Older sweeps included zero incorrectly: preserve evidence, exclude from
# fixed-seed reproducibility rates, and never interpret budget differences.
sub seed_outcome {
    my ($seed, $a, $b, $equal) = @_;
    return 'clock_seed_ungraded' if "$seed" eq '0';
    return outcome($a, $b, $equal);
}

sub engine_contrast {
    my ($local, $reference) = @_;
    return 'reference_not_run' unless ref($reference) eq 'HASH';
    return 'reference_error' if ($reference->{status} // '') eq 'response_error';
    return 'both_complete' if success($local) && success($reference);
    return 'local_only_failure' if !success($local) && success($reference);
    return 'reference_only_failure' if success($local) && !success($reference);
    return 'both_failed';
}

sub read_text {
    my ($path) = @_;
    open my $fh, '<:raw', $path or die "read $path: $!\n";
    local $/;
    return <$fh>;
}

sub read_json {
    my ($path) = @_;
    my $raw = eval { read_text($path) };
    return undef if $@;
    my $parsed = eval { decode_json($raw) };
    return ref($parsed) eq 'HASH' ? $parsed : undef;
}

sub comparable {
    my ($path) = @_;
    my $json = read_json($path);
    return undef unless $json && ref($json->{choices}) eq 'ARRAY';
    my $choice = $json->{choices}[0];
    return undef unless ref($choice) eq 'HASH' && ref($choice->{message}) eq 'HASH';
    return JSON::PP->new->canonical->encode({
        finish_reason => $choice->{finish_reason},
        reasoning_content => $choice->{message}{reasoning_content}
            // $choice->{message}{reasoning} // '',
        content => $choice->{message}{content} // '',
        tool_calls => [
            map {
                { type => $_->{type}, function => $_->{function} }
            } grep { ref($_) eq 'HASH' } @{$choice->{message}{tool_calls} // []}
        ],
    });
}

sub reason {
    my ($path) = @_;
    my $json = read_json($path);
    return '' unless $json;
    return $json->{choices}[0]{message}{reasoning_content}
        // $json->{choices}[0]{message}{reasoning} // '';
}

sub seed_report {
    my ($a, $b, $same, $ext, $long, $reference) = @_;
    my $status = outcome($a, $b, $same);
    my $replay = $status eq 'pass' ? 'not_needed'
        : success($ext) ? 'completed_at_16k'
        : success($long) ? 'completed_at_32k'
        : ref($long) eq 'HASH' ? 'still_failed_at_32k'
        : ref($ext) eq 'HASH' ? 'still_failed_at_16k'
        : 'not_run';
    return {
        status => $status, replay => $replay,
        baseline_a => $a, baseline_b => $b,
        extended_16k => $ext, extended_32k => $long,
        reference => engine_contrast($a, $reference),
    };
}

sub self_test {
    my $good = {status=>'ok', finish_reason=>'stop',
                final_answer_present=>JSON::PP::true};
    my $bad = {status=>'truncated', finish_reason=>'length',
               final_answer_present=>JSON::PP::false};
    my $repeat = {status=>'repetition', finish_reason=>'length',
                  final_answer_present=>JSON::PP::false,
                  reasoning_loop=>JSON::PP::true};
    my @tests = (
        [outcome($good,$good,1),'pass'],
        [outcome($good,$good,0),'nondeterministic'],
        [outcome($bad,$bad,1),'reproducible_failure'],
        [outcome($bad,$repeat,0),'varying_failure'],
        [outcome($bad,undef,0),'missing'],
        [seed_outcome(0,$bad,$bad,0),'clock_seed_ungraded'],
        [seed_outcome(1,$bad,$bad,1),'reproducible_failure'],
        [seed_report($bad,$bad,1,$good,undef,undef)->{replay},'completed_at_16k'],
        [seed_report($bad,$bad,1,$bad,$good,undef)->{replay},'completed_at_32k'],
        [seed_report($bad,$bad,1,$bad,$bad,undef)->{replay},'still_failed_at_32k'],
        [engine_contrast($bad,$good),'local_only_failure'],
        [engine_contrast($bad,$bad),'both_failed'],
        [engine_contrast($good,undef),'reference_not_run'],
        [engine_contrast($good,$bad),'reference_only_failure'],
    );
    for my $t (@tests) {
        die "self-test got $t->[0] expected $t->[1]\n" unless $t->[0] eq $t->[1];
    }
    print "seed-repro-report: ", scalar(@tests), " self-tests PASS\n";
}

if (@ARGV == 1 && $ARGV[0] eq '--self-test') {
    self_test();
    exit 0;
}

@ARGV == 1 or die "Usage: perl $0 RUN_DIRECTORY | --self-test\n";
my $out = $ARGV[0];
my $cases_file = read_text("$out/cases.jsonl");
my (%seeds, @cases);
for my $line (split /\n/, $cases_file) {
    next unless $line =~ /\S/;
    my $v = eval { decode_json($line) };
    die "Invalid cases.jsonl: $@\n" if $@ || ref($v) ne 'HASH';
    push @cases, $v;
    $seeds{$1} = 1 if ($v->{id} // '') =~ /^seed-([0-9]+)-a$/;
}
die "No baseline seeds in cases.jsonl\n" unless keys %seeds;

my $get = sub {
    my ($name, $side) = @_;
    return read_json("$out/$side/$name.analysis.json");
};
my $response = sub {
    my ($name, $side) = @_;
    return comparable("$out/$side/$name.response.json");
};
my @rows;
my $incomplete = 0;
my $failed = 0;
my $passed = 0;
my $unstable = 0;
my $ref_count = 0;
my $clock_excluded = 0;
for my $seed (sort { $a <=> $b } keys %seeds) {
    my $id = "seed-$seed";
    my $a = $get->("$id-a",'local');
    my $b = $get->("$id-b",'local');
    my $first = $response->("$id-a",'local');
    my $second = $response->("$id-b",'local');
    my $same = defined($first) && defined($second) && $first eq $second;
    my $ext = $get->("$id-16k",'local');
    my $long = $get->("$id-32k",'local');
    my $ref = $get->("$id-a",'reference');
    my $row = seed_report($a,$b,$same,$ext,$long,$ref);
    $row->{seed} = 0 + $seed;
    $row->{status} = seed_outcome($seed,$a,$b,$same);
    if ($row->{status} eq 'clock_seed_ungraded') {
        $row->{replay} = 'clock_seed_budget_incomparable';
        $row->{reference} = 'clock_seed_reference_incomparable';
    }
    $row->{same_response} = $same ? JSON::PP::true : JSON::PP::false;
    $row->{reference_complete} = success($ref) ? JSON::PP::true : JSON::PP::false;
    $ref_count++ if ref($ref) eq 'HASH';

    my $prefix = undef;
    if ($seed ne '0' && ref($a) eq 'HASH' && ($a->{finish_reason} // '') eq 'length') {
        my $short = reason("$out/local/$id-a.response.json");
        my $longer = reason("$out/local/$id-16k.response.json");
        if (length($short) && length($longer)) {
            $prefix = index($longer, $short) == 0 ? JSON::PP::true : JSON::PP::false;
        }
    }
    $row->{extended_prefix_match} = $prefix;
    if ($row->{status} eq 'clock_seed_ungraded') {
        $clock_excluded++;
    } elsif ($row->{status} eq 'pass') {
        $passed++;
    } elsif ($row->{status} eq 'missing') {
        $incomplete++;
    } else {
        $failed++;
        $unstable++ if !$same;
    }
    push @rows, $row;
}

my $greedy_local = $response->('greedy','local');
my $greedy_ref = $response->('greedy','reference');
my $greedy_match = defined($greedy_local) && defined($greedy_ref)
    ? ($greedy_local eq $greedy_ref ? JSON::PP::true : JSON::PP::false) : undef;
my $report = {
    verdict => $incomplete || ($passed + $failed == 0)
        ? 'INCONCLUSIVE' : $failed ? 'FAIL' : 'PASS',
    seeds_tested => scalar(@rows),
    fixed_seeds_tested => $passed + $failed + $incomplete,
    clock_seeds_excluded => $clock_excluded,
    passed => $passed, failed => $failed,
    incomplete => $incomplete, unstable => $unstable,
    reference_runs => $ref_count,
    greedy_reference_exact_match => $greedy_match,
    attribution => 'Seed 0 selects native clock entropy and is excluded; '
        . 'non-greedy seeds alone cannot prove engine or quantization defects; '
        . 'confirm identical weights, template, sampler and logit parity.',
    rows => \@rows,
};
my $json = JSON::PP->new->canonical->pretty->utf8->encode($report);
open my $j, '>:raw', "$out/report.json" or die "write report.json: $!";
print {$j} $json;
close $j;
open my $tsv, '>:encoding(UTF-8)', "$out/report.tsv" or die "write report.tsv: $!";
print {$tsv} "seed\tstatus\tsame_output\tbase_a\tbase_b\treplay\tprefix\tref\n";
for my $r (@rows) {
    my $prefix = !defined($r->{extended_prefix_match}) ? 'not_checked'
        : $r->{extended_prefix_match} ? 'MATCH' : 'MISMATCH';
    print {$tsv} join("\t", $r->{seed},$r->{status},
        $r->{status} eq 'clock_seed_ungraded' ? 'not_applicable'
            : $r->{same_response} ? 'yes' : 'no',
        $r->{baseline_a}{status} // 'missing',
        $r->{baseline_b}{status} // 'missing',
        $r->{replay},$prefix,$r->{reference}), "\n";
}
close $tsv;
print "seed-sweep verdict=$report->{verdict} pass=$passed fail=$failed "
    . "incomplete=$incomplete clock_excluded=$clock_excluded ref=$ref_count\n";
print "evidence: $out/report.tsv $out/report.json\n";
exit($incomplete || ($passed + $failed == 0) ? 2 : $failed ? 1 : 0);
