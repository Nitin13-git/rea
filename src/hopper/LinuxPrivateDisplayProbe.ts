import { randomUUID } from "node:crypto";

import {
  hopperStartupExitCode,
  type HopperPrivateDisplayStrategy,
  type HopperStartupDiagnostic,
  type HopperStartupFailureDiagnostic,
  type HopperStartupFailureCode,
  type HopperStartupReadyDiagnostic,
} from "../domain/hopperStartupFailure.js";
import { cleanupOwnedProcessGroup } from "../process/ProcessOwnership.js";
import {
  ProviderProcessSupervisor,
  spawnOwnedProviderProcess,
  type ProviderProcessLaunch,
  type ProviderProcessStopResult,
} from "../process/ProviderProcess.js";
import { parseLinuxPrivateDisplayDiagnostic } from "./LinuxPrivateDisplayDiagnostic.js";

const PYTHON_PATH = "/usr/bin/python3";
const UNSHARE_PATH = "/usr/bin/unshare";
const DEFAULT_PROBE_TIMEOUT_MS = 5_000;

const isAborted = (signal: AbortSignal | undefined): boolean =>
  signal?.aborted === true;

export type LinuxPrivateDisplayRunnableStrategy = Exclude<
  HopperPrivateDisplayStrategy,
  "unavailable"
>;

/** An owned probe whose stop was not verified; stopping it again retries cleanup. */
export interface RetainedLinuxPrivateDisplayProbe {
  stop(): Promise<ProviderProcessStopResult>;
}

export interface LinuxPrivateDisplayProbeProcessResult {
  readonly outcome: "exited" | "timed-out" | "cancelled" | "launch-failed";
  readonly exitCode: number | null | undefined;
  readonly stderr: string;
  readonly stderrBytes: number;
  /** Present only while the probe's owned process cleanup remains unverified. */
  readonly retainedProbe?: RetainedLinuxPrivateDisplayProbe;
}

export type LinuxPrivateDisplayProbeRunner = (
  strategy: LinuxPrivateDisplayRunnableStrategy,
  options: {
    readonly helperPath: string;
    readonly environment?: Readonly<NodeJS.ProcessEnv>;
    readonly timeoutMs: number;
    readonly signal?: AbortSignal;
  },
) => Promise<LinuxPrivateDisplayProbeProcessResult>;

export type LinuxPrivateDisplaySelection =
  | {
      readonly ok: true;
      readonly strategy: LinuxPrivateDisplayRunnableStrategy;
      readonly diagnostic: HopperStartupReadyDiagnostic;
    }
  | {
      readonly ok: false;
      readonly strategy: "unavailable";
      readonly exitCode: number;
      readonly diagnostic: HopperStartupFailureDiagnostic;
      /** Probes still owned after failed cleanup; the caller must retry their stop. */
      readonly retainedProbes: readonly RetainedLinuxPrivateDisplayProbe[];
    };

/** Probe the exact Xvfb boundary and select the least-privileged viable strategy. */
export const selectLinuxPrivateDisplayStrategy = async (options: {
  readonly helperPath: string;
  readonly environment?: Readonly<NodeJS.ProcessEnv>;
  readonly timeoutMs?: number;
  readonly signal?: AbortSignal;
  readonly runProbe?: LinuxPrivateDisplayProbeRunner;
}): Promise<LinuxPrivateDisplaySelection> => {
  const timeoutMs = options.timeoutMs ?? DEFAULT_PROBE_TIMEOUT_MS;
  const runner = options.runProbe ?? runLinuxPrivateDisplayProbe;
  const direct = await evaluatedProbe("direct", options, timeoutMs, runner);
  if (direct.ok) return direct;
  if (direct.diagnostic.failure_code !== "x11_socket_directory_unusable")
    return unavailable(direct.diagnostic, direct.retainedProbes);

  const isolated = await evaluatedProbe(
    "user-mount-namespace",
    options,
    timeoutMs,
    runner,
  );
  if (!isolated.ok)
    return unavailable(
      {
        ...direct.diagnostic,
        strategy: "unavailable",
        reason: "namespace_unavailable",
        fallback_reason: isolated.diagnostic.reason,
        effective_socket_directory_mode:
          isolated.diagnostic.effective_socket_directory_mode,
        effective_mount_read_only:
          isolated.diagnostic.effective_mount_read_only,
        xvfb_stderr_bytes: isolated.diagnostic.xvfb_stderr_bytes,
      },
      [...direct.retainedProbes, ...isolated.retainedProbes],
    );
  return {
    ok: true,
    strategy: "user-mount-namespace",
    diagnostic: {
      ...isolated.diagnostic,
      socket_directory_mode: direct.diagnostic.socket_directory_mode,
      mount_read_only: direct.diagnostic.mount_read_only,
      fallback_reason: null,
    },
  };
};

/** Run one helper probe in an owned process group and retain its full diagnostics. */
export const runLinuxPrivateDisplayProbe: LinuxPrivateDisplayProbeRunner =
  async (strategy, options) => {
    if (isAborted(options.signal)) return emptyProcessResult("cancelled");
    const command = privateDisplayProbeCommand(options.helperPath, strategy);
    let started: Awaited<ReturnType<typeof spawnOwnedProviderProcess>>;
    try {
      started = await spawnOwnedProviderProcess({
        command: command.command,
        arguments: command.arguments,
        runId: randomUUID(),
        expectedCommand: null,
        ...(options.environment === undefined
          ? {}
          : { hostEnvironment: options.environment }),
        ...(options.signal === undefined ? {} : { signal: options.signal }),
      });
    } catch (cause: unknown) {
      // best-effort cleanup: probe spawn failure is reported as launch-failed.
      void cause;
      return emptyProcessResult(
        isAborted(options.signal) ? "cancelled" : "launch-failed",
      );
    }
    const launch: ProviderProcessLaunch = {
      process: started.process,
      ownsProcessLifetime: true,
      cleanup: () => cleanupOwnedProcessGroup(started.ownership),
    };
    const supervisor = new ProviderProcessSupervisor(launch);
    const outcome = await waitForProbeOutputClose(
      supervisor,
      options.timeoutMs,
      options.signal,
    );
    let retained = false;
    if (outcome !== "exit") {
      const stopped = await supervisor.stop();
      retained = stopped.status === "incomplete";
    } else supervisor.dispose();
    const snapshot = supervisor.snapshot();
    return {
      outcome:
        outcome === "exit"
          ? "exited"
          : outcome === "timeout"
            ? "timed-out"
            : "cancelled",
      exitCode: snapshot.exitCode,
      stderr: snapshot.stderr.text,
      stderrBytes: snapshot.stderr.bytes,
      ...(retained ? { retainedProbe: supervisor } : {}),
    };
  };

const privateDisplayProbeCommand = (
  helperPath: string,
  strategy: LinuxPrivateDisplayRunnableStrategy,
): { readonly command: string; readonly arguments: readonly string[] } =>
  strategy === "direct"
    ? {
        command: PYTHON_PATH,
        arguments: [helperPath, "--probe", "--strategy", "direct"],
      }
    : {
        command: UNSHARE_PATH,
        arguments: [
          "--user",
          "--map-root-user",
          "--mount",
          "--propagation",
          "private",
          PYTHON_PATH,
          helperPath,
          "--probe",
          "--strategy",
          "user-mount-namespace",
          "--mount-private-x11",
        ],
      };

type ProbeVerdict =
  | {
      readonly ok: true;
      readonly strategy: LinuxPrivateDisplayRunnableStrategy;
      readonly diagnostic: HopperStartupReadyDiagnostic;
    }
  | {
      readonly ok: false;
      readonly diagnostic: HopperStartupFailureDiagnostic;
    };

type ProbeEvaluation =
  | Extract<ProbeVerdict, { readonly ok: true }>
  | (Extract<ProbeVerdict, { readonly ok: false }> & {
      readonly retainedProbes: readonly RetainedLinuxPrivateDisplayProbe[];
    });

const evaluatedProbe = async (
  strategy: LinuxPrivateDisplayRunnableStrategy,
  options: {
    readonly helperPath: string;
    readonly environment?: Readonly<NodeJS.ProcessEnv>;
    readonly signal?: AbortSignal;
  },
  timeoutMs: number,
  runner: LinuxPrivateDisplayProbeRunner,
): Promise<ProbeEvaluation> => {
  let processResult: LinuxPrivateDisplayProbeProcessResult;
  try {
    processResult = await runner(strategy, {
      helperPath: options.helperPath,
      ...(options.environment === undefined
        ? {}
        : { environment: options.environment }),
      timeoutMs,
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    });
  } catch (cause: unknown) {
    // Runner failure is reported with the synthetic probe_failed diagnostic.
    void cause;
    return {
      ok: false,
      diagnostic: syntheticDiagnostic(strategy, "probe_failed"),
      retainedProbes: [],
    };
  }
  const verdict = evaluateProbeResult(strategy, processResult);
  // Ownership of an unstopped probe outlives whatever diagnostic describes it.
  return verdict.ok
    ? verdict
    : {
        ...verdict,
        retainedProbes:
          processResult.retainedProbe === undefined
            ? []
            : [processResult.retainedProbe],
      };
};

const evaluateProbeResult = (
  strategy: LinuxPrivateDisplayRunnableStrategy,
  processResult: LinuxPrivateDisplayProbeProcessResult,
): ProbeVerdict => {
  if (processResult.outcome === "cancelled")
    return {
      ok: false,
      diagnostic: syntheticDiagnostic(strategy, "cancelled"),
    };
  if (processResult.outcome === "timed-out")
    return {
      ok: false,
      diagnostic: syntheticDiagnostic(strategy, "probe_timeout"),
    };
  if (processResult.outcome === "launch-failed")
    return {
      ok: false,
      diagnostic: syntheticDiagnostic(strategy, "probe_failed"),
    };
  const parsed = parseLinuxPrivateDisplayDiagnostic(processResult.stderr);
  if (!parsed.ok)
    return {
      ok: false,
      diagnostic: syntheticDiagnostic(strategy, parsed.reason),
    };
  const diagnostic = parsed.value;
  if (
    diagnostic.strategy !== strategy ||
    diagnostic.operation !== "probe" ||
    !exitMatchesDiagnostic(processResult.exitCode, diagnostic)
  )
    return {
      ok: false,
      diagnostic: syntheticDiagnostic(strategy, "diagnostic_malformed"),
    };
  return diagnostic.status === "ready"
    ? { ok: true, strategy, diagnostic }
    : { ok: false, diagnostic };
};

const exitMatchesDiagnostic = (
  exitCode: number | null | undefined,
  diagnostic: HopperStartupDiagnostic,
): boolean =>
  diagnostic.status === "ready"
    ? exitCode === 0
    : hopperStartupExitCode(diagnostic.failure_code) === exitCode;

const syntheticDiagnostic = (
  strategy: LinuxPrivateDisplayRunnableStrategy,
  reason: string,
  failureCode: HopperStartupFailureCode = "private_display_unavailable",
): HopperStartupFailureDiagnostic => ({
  component: "hopper_private_display",
  operation: "probe",
  status: "error",
  failure_code: failureCode,
  reason,
  socket_directory: "/tmp/.X11-unix",
  socket_directory_mode: null,
  mount_read_only: null,
  effective_socket_directory_mode: null,
  effective_mount_read_only: null,
  wsl: false,
  strategy,
  fallback_reason: null,
  xvfb_stderr_bytes: 0,
});

const unavailable = (
  diagnostic: HopperStartupFailureDiagnostic,
  retainedProbes: readonly RetainedLinuxPrivateDisplayProbe[],
): LinuxPrivateDisplaySelection => ({
  ok: false,
  strategy: "unavailable",
  exitCode: hopperStartupExitCode(diagnostic.failure_code) ?? 70,
  diagnostic: { ...diagnostic, strategy: "unavailable" },
  retainedProbes,
});

const emptyProcessResult = (
  outcome: Extract<
    LinuxPrivateDisplayProbeProcessResult["outcome"],
    "cancelled" | "launch-failed"
  >,
): LinuxPrivateDisplayProbeProcessResult => ({
  outcome,
  exitCode: undefined,
  stderr: "",
  stderrBytes: 0,
});

const waitForProbeOutputClose = async (
  supervisor: ProviderProcessSupervisor,
  timeoutMs: number,
  signal: AbortSignal | undefined,
): Promise<"exit" | "timeout" | "cancelled"> => {
  if (signal?.aborted === true) return "cancelled";
  let releaseAbort: (() => void) | undefined;
  const aborted = new Promise<"cancelled">((resolve) => {
    if (signal === undefined) return;
    const listener = (): void => resolve("cancelled");
    signal.addEventListener("abort", listener, { once: true });
    releaseAbort = () => signal.removeEventListener("abort", listener);
  });
  try {
    return await Promise.race([
      supervisor
        .waitForOutputClose(Math.max(1, timeoutMs))
        .then((closed): "exit" | "timeout" => (closed ? "exit" : "timeout")),
      aborted,
    ]);
  } finally {
    releaseAbort?.();
  }
};
