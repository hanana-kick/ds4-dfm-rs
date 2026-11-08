#!/usr/bin/env perl
# Classify bounded high-thinking termination without claiming an EOS root cause.
use strict;
use warnings;
use JSON::PP;
use Encode qw(decode);

my @profiles = qw(high-base high-extended low-control no-thinking);

sub complete {
    my ($r) = @_;
    return ($r->{status} // '') eq 'ok'
        && ($r->{finish_reason} // '') eq 'stop'
        && $r->{final_answer_present}
        && !$r->{reasoning_loop}
        && !$r->{answer_loop};
}

sub judge {
    my ($r, $http, $text) = @_;
    my $result = {
        verdict => 'INCONCLUSIVE', category => 'invalid_evidence',
        reason => 'Missing or failed request evidence',
        prefix_consistent => undef, profiles => {},
    };
    for my $id (@profiles) {
        my $v = $r->{$id};
        $result->{profiles}{$id} = $v if ref($v) eq 'HASH';
        return $result unless ref($v) eq 'HASH'
            && ($http->{$id} // '') eq '200'
            && ($v->{status} // '') ne 'response_error';
    }

    my ($short, $long) = @{$r}{qw(high-base high-extended)};
    if (($short->{truncated} || ($short->{finish_reason} // '') eq 'length')
        && length($text->{'high-base'} // '') > 0) {
        $result->{prefix_consistent} =
            index($text->{'high-extended'} // '', $text->{'high-base'}) == 0
                ? JSON::PP::true : JSON::PP::false;
    }

    if (!complete($r->{'low-control'}) || !complete($r->{'no-thinking'})) {
        @{$result}{qw(verdict category reason)} = (
            'INCONCLUSIVE', 'control_failed',
            'Low or thinking-off control did not finish with a final answer',
        );
    } elsif (grep { $r->{$_}{reasoning_loop} || $r->{$_}{answer_loop} } @profiles) {
        @{$result}{qw(verdict category reason)} = (
            'FAIL', 'repetition_detected',
            'A profile contains repeated 100-character windows',
        );
    } elsif (($long->{finish_reason} // '') eq 'length') {
        @{$result}{qw(verdict category reason)} = (
            'FAIL', 'high_still_truncated',
            'High thinking consumed the extended output budget; EOS handling is not proven faulty',
        );
    } elsif (($long->{finish_reason} // '') eq 'stop' && !$long->{final_answer_present}) {
        @{$result}{qw(verdict category reason)} = (
            'FAIL', 'thinking_only_stop',
            'High thinking stopped without a final answer; inspect DSML and EOS handling',
        );
    } elsif (!complete($long) || !$long->{reasoning_exposed}) {
        @{$result}{qw(verdict category reason)} = (
            'INCONCLUSIVE', 'high_unexpected_result',
            'High thinking did not produce a valid exposed-reasoning completion',
        );
    } elsif (($short->{finish_reason} // '') eq 'length'
             && !defined($result->{prefix_consistent})) {
        @{$result}{qw(verdict category reason)} = (
            'INCONCLUSIVE', 'missing_prefix',
            'Truncated high baseline has no exposed reasoning to compare',
        );
    } elsif (($short->{finish_reason} // '') eq 'length'
             && !$result->{prefix_consistent}) {
        @{$result}{qw(verdict category reason)} = (
            'INCONCLUSIVE', 'prefix_diverged',
            'High output differed before the short budget limit despite an identical seed',
        );
    } elsif (complete($short)) {
        @{$result}{qw(verdict category reason)} = (
            'PASS', 'high_completed_at_base',
            'High thinking completed within the initial budget',
        );
    } elsif (($short->{finish_reason} // '') eq 'length') {
        @{$result}{qw(verdict category reason)} = (
            'PASS', 'budget_resolved',
            'The identical seeded high reasoning prefix completed with the extended budget',
        );
    } else {
        @{$result}{qw(verdict category reason)} = (
            'INCONCLUSIVE', 'high_base_unexpected',
            'High baseline had an unexpected termination state',
        );
    }
    return $result;
}

sub read_file {
    my ($path) = @_;
    open my $f, '<:raw', $path or die "read $path: $!\n";
    local $/;
    return <$f>;
}

sub self_test {
    my $ok = sub {
        my ($finish, $answer, $extra) = @_;
        return { status => $finish eq 'length' ? 'truncated' : 'ok',
            finish_reason => $finish, final_answer_present => $answer ? JSON::PP::true : JSON::PP::false,
            truncated => $finish eq 'length' ? JSON::PP::true : JSON::PP::false,
            reasoning_exposed => JSON::PP::true, reasoning_loop => JSON::PP::false,
            answer_loop => JSON::PP::false, %{ $extra || {} } };
    };
    my %http = map { $_ => '200' } @profiles;
    my $base = sub { return {
        'high-base' => $ok->('length', 0),
        'high-extended' => $ok->('stop', 1),
        'low-control' => $ok->('stop', 1),
        'no-thinking' => $ok->('stop', 1),
    }};
    my %prefix = ('high-base'=>'a thought', 'high-extended'=>'a thought continued');
    my @cases;
    push @cases, ['budget_resolved', $base->(), \%http, \%prefix];
    my $trunc = $base->(); $trunc->{'high-extended'}=$ok->('length',0);
    push @cases, ['high_still_truncated', $trunc, \%http, \%prefix];
    my $empty = $base->(); $empty->{'high-extended'}=$ok->('stop',0);
    push @cases, ['thinking_only_stop', $empty, \%http, \%prefix];
    my $control = $base->(); $control->{'low-control'}=$ok->('length',0);
    push @cases, ['control_failed', $control, \%http, \%prefix];
    my $drift = $base->();
    push @cases, ['prefix_diverged', $drift, \%http,
        {'high-base'=>'a thought','high-extended'=>'another thought'}];
    my $loop = $base->(); $loop->{'high-extended'}=$ok->('stop',1,{reasoning_loop=>JSON::PP::true});
    push @cases, ['repetition_detected', $loop, \%http, \%prefix];
    my $fine = $base->(); $fine->{'high-base'}=$ok->('stop',1);
    push @cases, ['high_completed_at_base', $fine, \%http, \%prefix];
    my %broken = (%http, 'high-extended'=>'000');
    push @cases, ['invalid_evidence', $base->(), \%broken, \%prefix];
    for my $test (@cases) {
        my ($want,$r,$h,$texts)=@$test;
        my $got=judge($r,$h,$texts);
        die "verdict self-test expected $want got $got->{category}\n"
            unless $got->{category} eq $want;
    }
    print "thinking-verdict: ", scalar(@cases), " self-tests PASS\n";
}

if (@ARGV == 1 && $ARGV[0] eq '--self-test') {
    self_test();
    exit 0;
}
@ARGV == 1 or die "Usage: $0 RUN_DIRECTORY | --self-test\n";
my $dir = $ARGV[0];
my (%data, %http, %text);
for my $id (@profiles) {
    my $path = "$dir/c1-off/$id";
    my $json = eval { JSON::PP->new->utf8->decode(read_file("$path.analysis.json")) };
    $data{$id} = $json if !$@ && ref($json) eq 'HASH';
    $http{$id} = eval { read_file("$path.http") } // '';
    $http{$id} =~ s/\s+//g;
    $text{$id} = eval { decode('UTF-8', read_file("$path.thinking.txt"), 1) } // '';
    $text{$id} =~ s/\r?\n\z//;
}
my $v = judge(\%data, \%http, \%text);
print JSON::PP->new->canonical->pretty->utf8->encode($v);
exit($v->{verdict} eq 'PASS' ? 0 : $v->{verdict} eq 'FAIL' ? 1 : 2);
