#!/usr/bin/env python3
"""
MathClaw CLI - Command-line interface for autonomous mathematical discovery.

Usage:
    mathclaw start [--provider PROVIDER] [--model MODEL] [--config CONFIG]
    mathclaw stop
    mathclaw status
    mathclaw export [--format FORMAT] [--domain DOMAIN] [--output PATH]
    mathclaw discover [--count N] [--domain DOMAIN]
    mathclaw theorems [--domain DOMAIN] [--limit N]
    mathclaw config [--show | --create]

Examples:
    # Start autonomous discovery with OpenAI
    mathclaw start --provider openai
    
    # Run 5 discovery attempts
    mathclaw discover --count 5
    
    # Export theorems to markdown
    mathclaw export --format markdown --output discoveries.md
    
    # View discovered theorems
    mathclaw theorems --limit 20
"""

import sys
import time
from pathlib import Path

try:
    import click
except ImportError:
    print("Click not installed. Run: pip install click")
    sys.exit(1)


# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@click.group()
@click.version_option(version='1.0.0', prog_name='MathClaw')
def cli():
    """MathClaw - Autonomous Mathematical Discovery Engine"""
    pass


@cli.command()
@click.option('--provider', '-p', type=click.Choice(['openai', 'anthropic', 'gemini', 'ollama', 'auto']),
              default='auto', help='LLM provider to use')
@click.option('--model', '-m', help='Model to use (provider-specific)')
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to config file')
@click.option('--interval', '-i', type=int, default=5, help='Seconds between discovery attempts')
@click.option('--max-hours', type=float, default=None, help='Maximum hours to run')
def start(provider, model, config, interval, max_hours):
    """Start autonomous mathematical discovery."""
    from mathclaw.api import MathClawConfig, get_provider
    from mathclaw.discovery import DiscoveryEngine
    
    click.echo(click.style("🔮 MathClaw - Autonomous Mathematical Discovery", fg='cyan', bold=True))
    click.echo()
    
    # Load configuration
    if config:
        cfg = MathClawConfig.load(config_path=Path(config))
    else:
        cfg = MathClawConfig.load()
    
    # Override with CLI options
    if provider != 'auto':
        cfg.llm_provider = provider
    if model:
        cfg.llm_model = model
    if interval:
        cfg.min_interval_seconds = interval
    
    # Get LLM provider
    llm = get_provider(
        provider_type=cfg.llm_provider if cfg.llm_provider != 'auto' else None,
        model=cfg.llm_model,
    )
    
    if not llm:
        click.echo(click.style("❌ No LLM provider available!", fg='red'))
        click.echo("Please set an API key:")
        click.echo("  export OPENAI_API_KEY=sk-...")
        click.echo("  export ANTHROPIC_API_KEY=sk-ant-...")
        click.echo("  export GOOGLE_API_KEY=...")
        click.echo("Or start Ollama locally.")
        sys.exit(1)
    
    click.echo(f"📡 Provider: {llm.name}")
    click.echo(f"⏱️  Interval: {cfg.min_interval_seconds}s")
    click.echo(f"🎯 Max/hour: {cfg.max_attempts_per_hour}")
    click.echo()
    
    # Create engine
    engine = DiscoveryEngine(
        llm_provider=llm,
        config={
            'min_interval_seconds': cfg.min_interval_seconds,
            'max_attempts_per_hour': cfg.max_attempts_per_hour,
            'verification_timeout': cfg.verification_timeout,
        }
    )
    
    # Set up callbacks
    def on_discovery(conjecture, result):
        click.echo(click.style(f"✅ PROVEN: {conjecture.statement}", fg='green', bold=True))
    
    def on_attempt(conjecture):
        click.echo(f"🔍 Testing: {conjecture.statement[:60]}...")
    
    engine.on_discovery(on_discovery)
    engine.on_attempt(on_attempt)
    
    # Start
    click.echo(click.style("🚀 Starting autonomous discovery...", fg='yellow'))
    click.echo("Press Ctrl+C to stop")
    click.echo()
    
    start_time = time.time()
    max_seconds = max_hours * 3600 if max_hours else float('inf')
    
    try:
        engine.start()
        
        while engine.state.value == 'running':
            time.sleep(1)
            
            # Check time limit
            if time.time() - start_time > max_seconds:
                click.echo("\n⏰ Time limit reached")
                break
            
    except KeyboardInterrupt:
        click.echo("\n⏹️  Stopping...")
    finally:
        engine.stop()
    
    # Show final stats
    stats = engine.stats
    if stats:
        click.echo()
        click.echo(click.style("📊 Session Summary", fg='cyan', bold=True))
        click.echo(f"   Total attempts: {stats.total_attempts}")
        click.echo(f"   Proven: {stats.proven}")
        click.echo(f"   Disproven: {stats.disproven}")
        click.echo(f"   Success rate: {stats.success_rate:.1%}")
        click.echo(f"   Runtime: {stats.total_time_seconds:.1f}s")


@cli.command()
def status():
    """Show MathClaw status and statistics."""
    from mathclaw.discovery import TheoremStore
    from mathclaw.evolution import SuccessTracker, StrategyStore
    
    click.echo(click.style("📊 MathClaw Status", fg='cyan', bold=True))
    click.echo()
    
    # Theorem stats
    try:
        store = TheoremStore()
        stats = store.get_stats()
        
        click.echo(click.style("Theorem Database:", bold=True))
        click.echo(f"  Total theorems: {stats['total_theorems']}")
        
        if stats['by_domain']:
            click.echo("  By domain:")
            for domain, count in sorted(stats['by_domain'].items(), key=lambda x: -x[1]):
                click.echo(f"    {domain}: {count}")
        
        if stats['most_recent_discovery']:
            click.echo(f"  Most recent: {stats['most_recent_discovery'][:19]}")
    except Exception as e:
        click.echo(f"  (No data yet)")
    
    click.echo()
    
    # Discovery stats
    try:
        tracker = SuccessTracker()
        disc_stats = tracker.get_stats()
        
        click.echo(click.style("Discovery Statistics:", bold=True))
        click.echo(f"  Total attempts: {disc_stats['total_attempts']}")
        click.echo(f"  Success rate: {disc_stats['success_rate']:.1%}")
        click.echo(f"  Avg verification time: {disc_stats['avg_execution_time_ms']:.0f}ms")
    except Exception:
        click.echo(f"  (No data yet)")
    
    click.echo()
    
    # Strategy stats
    try:
        strategies = StrategyStore()
        all_strats = strategies.get_all_strategies()
        
        click.echo(click.style("Strategies:", bold=True))
        click.echo(f"  Total strategies: {len(all_strats)}")
        
        for s in sorted(all_strats, key=lambda x: -x.success_rate)[:5]:
            click.echo(f"    {s.id}: {s.success_rate:.1%} ({s.times_used} uses)")
    except Exception:
        click.echo(f"  (No data yet)")


@cli.command()
@click.option('--count', '-n', type=int, default=1, help='Number of discovery attempts')
@click.option('--domain', '-d', help='Mathematical domain to focus on')
@click.option('--provider', '-p', type=click.Choice(['openai', 'anthropic', 'gemini', 'ollama', 'auto']),
              default='auto', help='LLM provider')
def discover(count, domain, provider):
    """Run discovery attempts (non-autonomous mode)."""
    from mathclaw.api import MathClawConfig, get_provider
    from mathclaw.discovery import DiscoveryEngine
    
    cfg = MathClawConfig.load()
    
    llm = get_provider(
        provider_type=provider if provider != 'auto' else None,
    )
    
    if not llm:
        click.echo(click.style("❌ No LLM provider available!", fg='red'))
        sys.exit(1)
    
    engine = DiscoveryEngine(llm_provider=llm)
    
    click.echo(f"🔍 Running {count} discovery attempt(s)...")
    click.echo()
    
    proven = 0
    for i in range(count):
        result = engine.run_once()
        
        status_icon = {
            'proven': '✅',
            'disproven': '❌',
            'plausible': '🟡',
            'unknown': '❓',
            'error': '⚠️',
        }.get(result.get('verification_status', 'error'), '❓')
        
        click.echo(f"{i+1}. {status_icon} {result.get('conjecture', 'Error')[:60]}")
        
        if result.get('verification_status') == 'proven':
            proven += 1
            click.echo(click.style(f"   ✨ Proven via {result.get('proof_method')}", fg='green'))
    
    click.echo()
    click.echo(f"Results: {proven}/{count} proven")


@cli.command('export')
@click.option('--format', '-f', type=click.Choice(['markdown', 'latex', 'json', 'jupyter']),
              default='markdown', help='Export format')
@click.option('--domain', '-d', help='Filter by domain')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
def export_cmd(format, domain, output):
    """Export discovered theorems."""
    from mathclaw.discovery import TheoremStore, KnowledgeExporter
    
    store = TheoremStore()
    exporter = KnowledgeExporter(store)
    
    count = store.count()
    click.echo(f"📦 Exporting {count} theorems as {format}...")
    
    if output is None:
        output = f"mathclaw_theorems.{format if format != 'jupyter' else 'ipynb'}"
    
    output_path = Path(output)
    
    if format == 'markdown':
        exporter.to_markdown(output_path, domain=domain)
    elif format == 'latex':
        exporter.to_latex(output_path, domain=domain)
    elif format == 'json':
        exporter.to_json(output_path, domain=domain)
    elif format == 'jupyter':
        exporter.to_jupyter(output_path, domain=domain)
    
    click.echo(click.style(f"✅ Exported to {output_path}", fg='green'))


@cli.command()
@click.option('--domain', '-d', help='Filter by domain')
@click.option('--limit', '-l', type=int, default=20, help='Maximum theorems to show')
def theorems(domain, limit):
    """List discovered theorems."""
    from mathclaw.discovery import TheoremStore
    
    store = TheoremStore()
    
    if domain:
        results = store.get_by_domain(domain, limit=limit)
    else:
        results = store.get_recent(limit=limit)
    
    if not results:
        click.echo("No theorems discovered yet.")
        click.echo("Run 'mathclaw discover' or 'mathclaw start' to begin.")
        return
    
    click.echo(click.style(f"📚 Discovered Theorems ({len(results)} shown)", fg='cyan', bold=True))
    click.echo()
    
    for i, t in enumerate(results, 1):
        click.echo(click.style(f"{i}. {t.statement}", bold=True))
        if t.natural_language != t.statement:
            click.echo(f"   {t.natural_language}")
        click.echo(f"   Domain: {t.domain} | Method: {t.proof_method} | Date: {t.discovered_at[:10]}")
        click.echo()


@cli.command()
@click.option('--show', is_flag=True, help='Show current configuration')
@click.option('--create', is_flag=True, help='Create example config files')
def config(show, create):
    """Manage configuration."""
    from mathclaw.api import MathClawConfig
    
    if create:
        cfg = MathClawConfig()
        cfg.save()
        cfg.create_example_env()
        
        click.echo(click.style("✅ Created configuration files:", fg='green'))
        click.echo("  mathclaw.yaml - Main config")
        click.echo("  .env.example - API keys template")
        click.echo()
        click.echo("Copy .env.example to .env and add your API keys.")
        return
    
    # Default: show config
    cfg = MathClawConfig.load()
    
    click.echo(click.style("⚙️  MathClaw Configuration", fg='cyan', bold=True))
    click.echo()
    
    for key, value in cfg.to_dict().items():
        click.echo(f"  {key}: {value}")


@cli.command()
def health():
    """Run system health checks."""
    from mathclaw.protection import HealthChecker
    
    click.echo(click.style("🏥 Running health checks...", fg='cyan'))
    click.echo()
    
    try:
        checker = HealthChecker()
        report = checker.run_full_check()
        
        click.echo(click.style(f"Status: {report['status'].upper()}", 
                              fg='green' if report['status'] == 'healthy' else 'red',
                              bold=True))
        click.echo()
        
        for check in report['checks']:
            icon = '✅' if check['passed'] else '❌'
            click.echo(f"  {icon} {check['name']}: {check['message']}")
        
        if report['status'] != 'healthy':
            click.echo()
            click.echo(click.style("⚠️  Some checks failed!", fg='yellow'))
            click.echo(f"Recommendation: {report.get('recommendation', 'Review failed checks')}")
            
    except Exception as e:
        click.echo(click.style(f"❌ Health check failed: {e}", fg='red'))


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
